"""Separate radiology-text audit. Text never enters the main benchmark."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import false_discovery_control
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from .baselines import P,load_data
from .metrics import score,select_threshold
from .aggregate import align_patients
from .bootstrap import paired_bootstrap

SECTIONS=('EXAMINATION','INDICATION','TECHNIQUE','COMPARISON','FINDINGS','IMPRESSION')
AUDIT_VERSION='inline_headers_v2'
HEADER=re.compile(r'\b('+'|'.join(SECTIONS)+r')\s*:\s*',re.I)
LEXICON={'death':r'\bdeath\b','deceased':r'\bdeceased\b','expired':r'\bexpired\b',
         'post_mortem':r'\bpost[-\s]?mortem\b','brain_death':r'\bbrain\s+death\b',
         'comfort_care':r'\bcomfort\s+care\b','dnr_dni':r'\b(?:dnr|dni)\b',
         'withdrawal_of_care':r'\bwithdrawal\s+of\s+(?:care|life\s+support)\b','hospice':r'\bhospice\b'}
FLAGGED=re.compile('|'.join('(?:'+pattern+')' for pattern in LEXICON.values()),re.I)
TOKEN=r'(?u)\b[a-zA-Z]{2,}\b'


def normalize_report(text):
    text='' if pd.isna(text) else str(text)
    listed=False
    if text.lstrip().startswith('['):
        try:
            parsed=ast.literal_eval(text)
            if isinstance(parsed,list) and all(isinstance(item,str) for item in parsed):
                text='\n'.join(parsed)
                listed=True
        except (ValueError,SyntaxError):
            pass
    text=text.replace('\\n','\n')
    text=re.sub(r'\[\*\*.*?\*\*\]',' ',text,flags=re.S)
    return text,listed


def split_sections(text):
    matches=list(HEADER.finditer(text))
    sections={name:[] for name in SECTIONS}
    for index,match in enumerate(matches):
        end=matches[index+1].start() if index+1<len(matches) else len(text)
        sections[match.group(1).upper()].append(text[match.end():end])
    return {name:'\n'.join(parts) for name,parts in sections.items()},len(matches)>len({match.group(1).upper() for match in matches})


def log_odds(texts,labels):
    vectorizer=CountVectorizer(ngram_range=(1,2),min_df=10,max_features=30000,token_pattern=TOKEN,lowercase=True)
    try:
        matrix=vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame()
    if matrix.shape[1]<2:
        return pd.DataFrame()
    labels=np.asarray(labels)
    deceased=np.asarray(matrix[labels==1].sum(axis=0)).ravel().astype(float)
    surviving=np.asarray(matrix[labels==0].sum(axis=0)).ravel().astype(float)
    pooled=deceased+surviving
    # Fixed empirical-Bayes prior: strength 1,000; shape pooled TRAINING counts.
    alpha=1000.*pooled/pooled.sum()
    d=deceased+alpha
    s=surviving+alpha
    delta=np.log(d/np.maximum(deceased.sum()+1000.-d,1e-12))-np.log(s/np.maximum(surviving.sum()+1000.-s,1e-12))
    z=delta/np.sqrt(1./d+1./s)
    return pd.DataFrame({'term':vectorizer.get_feature_names_out(),'log_odds':delta,'z':z,
                         'p':2*ndtr(-np.abs(z)),'deceased_tokens':deceased,'surviving_tokens':surviving})


def predictive_audit(texts,frame,parts,section,strip,output):
    experiment=f'report_v2__{section.lower()}__'+('stripped' if strip else 'original')
    directory=output/experiment
    directory.mkdir(exist_ok=True)
    texts=np.asarray([FLAGGED.sub(' ',text) if strip else text for text in texts],dtype=object)
    y=frame.label.to_numpy()
    rows=[]
    featureless=0
    for fold in range(5):
        path=directory/f'fold{fold}.parquet'
        if path.exists():
            previous=pd.read_parquet(path)
            featureless+=int(previous.featureless_fold.iloc[0])
            rows.append(previous)
            continue
        train,val,test=[parts[f'fold{fold}_{key}'] for key in ('train','validation','test')]
        vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=10,max_features=30000,token_pattern=TOKEN,lowercase=True)
        try:
            x=vectorizer.fit_transform(texts[train])
        except ValueError:
            featureless+=1
            validation=np.full(len(val),y[train].mean())
            probability=np.full(len(test),y[train].mean())
            model=None
        else:
            model=LogisticRegression(C=1.,penalty='l2',solver='liblinear',max_iter=2000,random_state=0)
            model.fit(x,y[train])
            validation=model.predict_proba(vectorizer.transform(texts[val]))[:,1]
            probability=model.predict_proba(vectorizer.transform(texts[test]))[:,1]
        threshold=select_threshold(y[val],validation)
        result=frame.iloc[test][['stay_id','subject_id','fold']].copy()
        result['y_true']=y[test]
        result['y_prob']=probability
        result['threshold']=threshold
        result['featureless_fold']=model is None
        joblib.dump({'vectorizer':vectorizer,'model':model,'featureless':model is None},directory/f'fold{fold}.joblib')
        temporary=path.with_suffix('.parquet.part')
        result.to_parquet(temporary,index=False)
        temporary.replace(path)
        rows.append(result)
    oof=pd.concat(rows).sort_values('subject_id')
    if len(oof)!=len(frame) or oof.subject_id.duplicated().any():
        raise RuntimeError('Report audit OOF patient coverage mismatch')
    oof.to_parquet(P/f'results/oof/{experiment}__seed0.parquet',index=False)
    return oof,{'section':section,'strip_flagged_terms':strip,'featureless_folds':featureless,
                **score(oof.y_true,oof.y_prob,oof.threshold.to_numpy())}


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Report audit requires a CPU Slurm allocation')
    data=load_data('main')
    frame,parts=data['frame'],data['parts']
    output=P/'results/report_audit'/AUDIT_VERSION
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'signature.txt'
    digest=hashlib.sha256(Path(__file__).read_bytes()+(data['directory']/'partitions.npz').read_bytes()).hexdigest()
    if lock.exists() and lock.read_text().strip()!=digest:
        raise RuntimeError('Report audit implementation changed after results were locked')
    lock.write_text(digest+'\n')
    manifest=json.loads((P/'results/acquisition/cxrmetadata.json').read_text())
    original=pd.concat([pd.read_csv(P/f'raw/mimiccxr/metadata{i}.csv',usecols=['subject_id','text']) for i in range(len(manifest))])
    if original.subject_id.duplicated().any():
        raise RuntimeError('Duplicate patient text rows')
    original=original.set_index('subject_id').loc[frame.subject_id,'text']
    parsed={name:[] for name in SECTIONS}
    repeated=listed=0
    for raw in original:
        text,is_list=normalize_report(raw)
        sections,has_repeated=split_sections(text)
        repeated+=int(has_repeated)
        listed+=int(is_list)
        for name in SECTIONS:
            parsed[name].append(sections[name])
    training=parts['fold0_train']
    training_y=frame.label.to_numpy()[training]
    term_tables=[]
    parse_rates={name:float(np.mean([bool(text.strip()) for text in texts])) for name,texts in parsed.items()}
    lexicon_rows=[]
    predictive=[]
    image_path=P/'results/oof/main__cxr__logistic__counts0__selectionmain__seed0.parquet'
    image=pd.read_parquet(image_path) if image_path.exists() else None
    for section,texts in parsed.items():
        terms=log_odds([texts[index] for index in training],training_y)
        if not terms.empty:
            terms['section']=section
            term_tables.append(terms)
        for label in (0,1):
            selected=[texts[index] for index in np.flatnonzero(frame.label.to_numpy()==label)]
            for term,pattern in LEXICON.items():
                matched=sum(bool(re.search(pattern,text,re.I)) for text in selected)
                lexicon_rows.append({'section':section,'label':label,'term':term,'reports':len(selected),
                                     'matching_reports':matched,'fraction':matched/len(selected)})
        for strip in (False,True):
            prediction,record=predictive_audit(texts,frame,parts,section,strip,output)
            if image is not None:
                paired=align_patients(prediction,image)
                difference=paired_bootstrap(paired.y_true_first,paired.subject_id,paired.y_prob_first,paired.y_prob_second)
                record['versus_image']=difference
            predictive.append(record)
            print('REPORT_AUDIT_SECTION_COMPLETE',section,'stripped',strip,flush=True)
    pd.DataFrame(lexicon_rows).to_csv(output/'lexicon_rates.csv',index=False)
    if term_tables:
        terms=pd.concat(term_tables,ignore_index=True)
        terms['q']=false_discovery_control(terms.p.to_numpy(),method='bh')
        terms.to_csv(output/'training_log_odds.csv',index=False)
        figures=P/'reports/figures/report_audit'/AUDIT_VERSION
        figures.mkdir(parents=True,exist_ok=True)
        for section in SECTIONS:
            for label,sign in [('deceased',1),('surviving',-1)]:
                subset=terms.loc[(terms.section==section)&(terms.q<.05)&(terms.log_odds*sign>0)]
                weights=dict(zip(subset.term,(subset.log_odds*sign).to_numpy()))
                fig,ax=plt.subplots(figsize=(10,5))
                if weights:
                    cloud=WordCloud(width=1200,height=600,background_color='white',max_words=100,random_state=0).generate_from_frequencies(weights)
                    ax.imshow(cloud,interpolation='bilinear')
                else:
                    ax.text(.5,.5,'No terms pass the fixed q < 0.05 rule',ha='center',va='center')
                ax.axis('off')
                ax.set_title(f'{section}: {label}-associated words/phrases')
                fig.text(.5,.01,'Fold-0 inner training patients only; log-odds magnitude weights; global BH q < 0.05.\nDescriptive association, not modality importance; report timestamps are unknown.',ha='center',fontsize=9)
                fig.tight_layout(rect=(0,.06,1,1))
                fig.savefig(figures/f'{section.lower()}_{label}.png',dpi=150,bbox_inches='tight')
                plt.close(fig)
    report={'audit_version':AUDIT_VERSION,'parse_rates':parse_rates,'patients':len(frame),'repeated_section_header_patients':repeated,
            'list_encoded_report_patients':listed,'predictive':predictive,
            'association_population':'fold0 inner training only; min patient document frequency 10',
            'prior':'empirical Bayes pooled training proportions; fixed strength 1000',
            'verdict':'NOT_ELIGIBLE_FOR_MAIN_BENCHMARK_OR_EXTRA_INPUT_ARM: timing inside 48h cannot be established',
            'augmentation':'Not modeled; mechanism and provenance remain unverified',
            'section_lexicon_denominator':'all patients in the outcome class, including absent sections'}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print('REPORT_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        reason=str(error) if isinstance(error,RuntimeError) else type(error).__name__
        (P/'reports/BLOCKED.md').write_text(f'Report audit stopped: {reason}. No automatic retries.\n')
        print('REPORT_AUDIT_STOPPED',reason,flush=True)
        raise SystemExit(1)
