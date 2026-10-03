"""Derive conservative common-field rules from official layouts and codebooks."""
import html
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[2]
DOWNLOADS=ROOT/'artifacts/runs/large_native_raw'
LAYOUT_FILES={year:ROOT/f'configs/brfss_layout_{year}.json' for year in (2024,2025)}
NAMES=['SEXVAR','GENHLTH','PHYSHLTH','MENTHLTH','POORHLTH','PRIMINS2','PERSDOC3','MEDCOST1',
       'CHECKUP1','EXERANY2','CVDINFR4','CVDCRHD4','CVDSTRK3','CHCSCNC1','CHCOCNC1','CHCCOPD3',
       'ADDEPEV3','CHCKDNY2','HAVARTH4','MARITAL','EDUCA','RENTHOM1','VETERAN3','EMPLOY1',
       'CHILDREN','INCOME3','DEAF','BLIND','DECIDE','DIFFWALK','DIFFDRES','DIFFALON',
       'SMOKE100','SMOKDAY2','USENOW3','ECIGNOW3','ALCDAY4','_AGE80','_BMI5']


def clean(value):
    return re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',value))).strip()


def source_rules(year):
    text=LAYOUT_FILES[year].read_text(encoding='utf-8')
    rows=json.loads(text)
    layout={row[1]:dict(start=int(row[0]),width=int(row[2])) for row in rows if row[0].isdigit()}
    with zipfile.ZipFile(DOWNLOADS/f'codebook{str(year)[-2:]}_llcp-v2-508.zip') as archive:
        member=next(name for name in archive.namelist() if name.lower().endswith('.html'))
        raw=archive.read(member)
        try:
            text=raw.decode('utf-8')
        except UnicodeDecodeError:
            text=raw.decode('cp1252')
    definitions={}
    for table in re.findall(r'<table\b.*?</table>',text,re.S|re.I):
        variable=re.search(r'SAS Variable Name:\s*([A-Z_0-9]+)',clean(table))
        if not variable:
            continue
        entries=[]
        for row in re.findall(r'<tr\b.*?</tr>',table,re.S|re.I):
            cells=[clean(cell) for cell in re.findall(r'<td\b[^>]*>(.*?)</td>',row,re.S|re.I)]
            if len(cells)<2 or not re.fullmatch(r'\d+(?:\s*[-–—]\s*\d+)?',cells[0]):
                continue
            numbers=list(map(int,re.findall(r'\d+',cells[0])))
            codes=list(range(numbers[0],numbers[-1]+1))
            entries.append(dict(codes=codes,label=cells[1]))
        definitions.setdefault(variable.group(1),[]).extend(entries)
    return layout,definitions


if __name__=='__main__':
    sources={year:source_rules(year) for year in (2024,2025)}
    variables=[]
    for name in NAMES+['DIABETE4','ASTHMA3','_STATE']:
        values=[]
        labels={}
        for year,(layout,definitions) in sources.items():
            if name not in layout or name not in definitions:
                raise ValueError(f'Missing official field {name} in {year}')
            valid=set()
            for entry in definitions[name]:
                label=entry['label']
                labels[str(year)+':'+str(entry['codes'][0])]=label
                if not re.search(r"don.t know|not sure|refused|missing|not asked",label,re.I):
                    valid.update(entry['codes'])
            values.append(valid)
        valid=sorted(set.intersection(*values))
        rule=dict(name=name,layouts={str(year):source[0][name] for year,source in sources.items()},
                  valid_values=valid,remap={},codebook_labels=labels)
        if name in ('PHYSHLTH','MENTHLTH','POORHLTH','CHILDREN') and 88 in valid:
            rule['remap']={'88':0}
        if name=='ALCDAY4' and 888 in valid:
            rule['remap']={'888':0}
        if name=='DIABETE4':
            rule.update(valid_values=[1,3],positive_value=1,negative_value=3)
            diabetes=rule
        elif name=='ASTHMA3':
            rule.update(valid_values=[1,2],positive_value=1,negative_value=2)
            asthma=rule
        elif name=='_STATE':
            rule['valid_values']=sorted(set.union(*values))
            group=rule
        else:
            if not valid:
                raise ValueError('Empty common code set: '+name)
            variables.append(rule)
    schema=dict(variables=variables,targets=[diabetes,asthma],group=group,
                source_urls=[f'https://www.cdc.gov/brfss/annual_data/{year}/llcp_varlayout_{str(year)[-2:]}_onecolumn.html' for year in (2024,2025)],
                missing_policy='Blank or outside common documented valid codes becomes NaN; legitimate insurance/income codes 7/9 retained. Zero remaps are explicit.',
                exclusions='No state/row identifiers, weights, outcome-specific modules, asthma-current status, diabetes age/type/care or outcome-derived calculations.',
                exceptions='_AGE80 and _BMI5 are demographic/anthropometric covariates, not outcome-derived quantities. The published _AGE80 field is imputed.',
                limitations='Cross-sectional self-reports; retrospective classification only. Survey design and state composition differ; respondents cannot be linked across years.')
    path=ROOT/'configs/brfss_common_variables_v1.json'
    path.write_text(json.dumps(schema,indent=2),encoding='utf-8')
    print(json.dumps({v['name']:[len(v['valid_values']),min(v['valid_values']),max(v['valid_values'])] for v in variables}))
