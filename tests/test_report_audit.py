from icu_mm.report_audit import normalize_report,split_sections,FLAGGED


def test_report_list_sections_and_deidentification_are_handled_without_execution():
    text,listed=normalize_report("['INDICATION: [**redacted**] cough\nFINDINGS: clear', 'INDICATION: fever\nIMPRESSION: opacity']".replace('\n','\\n'))
    sections,repeated=split_sections(text)
    assert listed and repeated
    assert 'redacted' not in text
    assert 'cough' in sections['INDICATION'] and 'fever' in sections['INDICATION']
    assert sections['IMPRESSION']=='opacity'
    assert FLAGGED.sub(' ','Comfort care and hospice.').lower().strip()=='and  .'
