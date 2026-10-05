"""Transport original HAR archive; audit subjects/features without opening labels."""
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/har_label_closed_v1'
OUT = ROOT / 'artifacts/manifests/har_label_closed_audit.json'
FREEZE = ROOT / 'artifacts/manifests/har_label_closed_freeze.json'
URL = 'https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip'


def sha(b): return hashlib.sha256(b).hexdigest()


def main():
    start=time.monotonic()
    for path,h in json.loads(FREEZE.read_text()).items():
        assert sha((ROOT/path).read_bytes())==h,path
    RAW.mkdir(parents=True,exist_ok=False)
    archive=RAW/'uci240.zip'
    with urllib.request.urlopen(URL,timeout=90) as response, archive.open('wb') as f:
        total=0
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            total+=len(chunk);assert total<=90*1024*1024,'archive bound'
            f.write(chunk)
    blob=archive.read_bytes(); opened=[];files={}

    def read(z,name):
        assert name.endswith(('README.txt','features_info.txt','features.txt','subject_train.txt','subject_test.txt','X_train.txt','X_test.txt')),name
        opened.append(name);b=z.read(name);files[name]=dict(sha256=sha(b),bytes=len(b));return b

    outer=zipfile.ZipFile(io.BytesIO(blob))
    names=outer.namelist()
    nested=[n for n in names if n.endswith('UCI HAR Dataset.zip')]
    z=zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
    assert not nested or len(nested)==1
    prefix='UCI HAR Dataset/'
    members=set(z.namelist())
    for name in ('README.txt','features_info.txt','features.txt'):
        b=read(z,prefix+name)
        (RAW/name).write_bytes(b)
    schema=(RAW/'features.txt').read_text().splitlines();assert len(schema)==561
    features={};subjects={}
    for split,number in [('train',7352),('test',2947)]:
        subjects[split]=np.loadtxt(io.BytesIO(read(z,prefix+split+'/subject_'+split+'.txt')),dtype=np.int64)
        features[split]=np.loadtxt(io.BytesIO(read(z,prefix+split+'/X_'+split+'.txt')),dtype=np.float64)
        assert subjects[split].shape==(number,)
        assert features[split].shape==(number,561)
        assert np.isfinite(features[split]).all()
        assert ((subjects[split]>=1)&(subjects[split]<=30)).all()
    train_ids=set(map(int,np.unique(subjects['train'])));test_ids=set(map(int,np.unique(subjects['test'])))
    assert len(train_ids)==21 and len(test_ids)==9 and not train_ids & test_ids
    order=sorted(train_ids,key=lambda i:sha(f'mira-har-source-v1:{i}'.encode()))
    allocation={'source_fit':order[:14],'source_development':order[14:],'evaluation_label_closed':sorted(test_ids)}
    fingerprints={}
    duplicates={}
    for group,ids in allocation.items():
        split='test' if group=='evaluation_label_closed' else 'train'
        rows=features[split][np.isin(subjects[split],ids)]
        digests=[sha(np.asarray(row,dtype='<f8').tobytes()) for row in rows]
        fingerprints[group]=set(digests)
        duplicates[group]=len(digests)-len(set(digests))
    overlap={}
    for a,b in [('source_fit','source_development'),('source_fit','evaluation_label_closed'),('source_development','evaluation_label_closed')]:
        overlap[a+'__'+b]=len(fingerprints[a]&fingerprints[b])
    label_members=[n for n in members if n.endswith(('/y_train.txt','/y_test.txt'))]
    assert len(label_members)==2 and not set(opened)&set(label_members)
    result=dict(date='2026-10-05',url=URL,doi='10.24432/C54S4K',license='CC BY 4.0',
        archive_sha256=sha(blob),archive_bytes=len(blob),nested_archive=bool(nested),
        member_reads=opened,opened_member_hashes=files,activity_labels_parsed=False,
        compressed_labels_transported=True,models_trained=0,cloud_calls=0,
        source_train_rows=7352,evaluation_rows=2947,features=561,subject_allocation=allocation,
        subject_rows={str(i):int((subjects['train']==i).sum()+(subjects['test']==i).sum()) for i in range(1,31)},
        duplicates_within=duplicates,duplicate_intersections=overlap,
        subject_boundary_passed=True,exact_duplicate_boundary_passed=not any(overlap.values()),
        source_readme_sha256=sha((RAW/'README.txt').read_bytes()),seconds=time.monotonic()-start,
        eligibility='Whole-subject classification only; mechanism/evaluation protocol not approved by this audit',
        limitations=['Windows overlap within subjects; no temporal identities reconstructed',
                    'No native missingness; future masks/costs must be declared synthetic',
                    'Repository access search cannot exclude undocumented or pretrained-model access'])
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('archive_sha256','subject_allocation','duplicate_intersections','subject_boundary_passed','exact_duplicate_boundary_passed','activity_labels_parsed','seconds')}))


if __name__=='__main__':main()
