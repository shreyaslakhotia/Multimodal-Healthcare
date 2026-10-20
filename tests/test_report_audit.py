from icu_mm.report_audit import normalize_report,split_sections,FLAGGED


def test_report_list_sections_and_deidentification_are_handled_without_execution():
    text,listed=normalize_report("['INDICATION: [**redacted**] cough\nFINDINGS: clear', 'INDICATION: fever\nIMPRESSION: opacity']".replace('\n','\\n'))
    sections,repeated=split_sections(text)
    assert listed and repeated
    assert 'redacted' not in text
    assert 'cough' in sections['INDICATION'] and 'fever' in sections['INDICATION']
    assert sections['IMPRESSION']=='opacity'
    assert FLAGGED.sub(' ','Comfort care and hospice.').lower().strip()=='and  .'


def test_inline_headers_are_split_without_assuming_original_newlines():
    sections,repeated=split_sections('FINDINGS: clear. IMPRESSION: normal. INDICATION: cough.')
    assert sections['FINDINGS'].strip()=='clear.'
    assert sections['IMPRESSION'].strip()=='normal.'
    assert sections['INDICATION'].strip()=='cough.'
    assert not repeated
