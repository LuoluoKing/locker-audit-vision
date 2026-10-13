# -*- coding: utf-8 -*-
"""阶段6 文献核验工具（OpenAlex 主力 + arXiv 补充）。

纪律：不预填任何元数据；所有候选必须来自真实 API 响应。
用法：python tools/verify_citations.py [group]
  group=all(默认) / 1 / 2
"""
import os, sys, json, time, urllib.request, urllib.parse, socket, re

for k in list(os.environ):
    if 'proxy' in k.lower():
        del os.environ[k]
socket.setdefaulttimeout(25)

UA = {'User-Agent': 'Mozilla/5.0 (citation-verification; mailto:research@example.org)'}


def openalex(query, rows=4):
    url = ('https://api.openalex.org/works?search=' + urllib.parse.quote(query)
           + f'&per-page={rows}&mailto=research@example.org')
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.loads(r.read().decode('utf-8'))
        out = []
        for w in d.get('results', []):
            au = [a.get('author', {}).get('display_name', '?')
                  for a in (w.get('authorships') or [])][:6]
            src = ((w.get('primary_location') or {}).get('source') or {})
            bib = w.get('biblio') or {}
            out.append({
                'title': w.get('title') or '?',
                'authors': au,
                'year': w.get('publication_year'),
                'venue': src.get('display_name') or '',
                'type': w.get('type') or '',
                'doi': (w.get('doi') or '').replace('https://doi.org/', ''),
                'pages': f"{bib.get('first_page') or ''}-{bib.get('last_page') or ''}".strip('-'),
                'volume': bib.get('volume') or '',
                'cited': w.get('cited_by_count', 0),
            })
        return out
    except Exception as e:
        return [{'error': f'{type(e).__name__}: {str(e)[:100]}'}]


def arxiv(query, rows=4):
    url = ('http://export.arxiv.org/api/query?search_query='
           + urllib.parse.quote(query) + f'&max_results={rows}')
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=25) as r:
            xml = r.read().decode('utf-8')
        entries = re.findall(r'<entry>(.*?)</entry>', xml, re.S)
        out = []
        for e in entries:
            t = re.search(r'<title>(.*?)</title>', e, re.S)
            au = re.findall(r'<name>(.*?)</name>', e)
            pub = re.search(r'<published>(\d{4})', e)
            aid = re.search(r'<id>(.*?)</id>', e)
            out.append({
                'title': re.sub(r'\s+', ' ', t.group(1)).strip() if t else '?',
                'authors': au[:6],
                'year': pub.group(1) if pub else '?',
                'arxiv_id': aid.group(1) if aid else '',
                'cited': '',
                'venue': 'arXiv', 'type': 'preprint', 'doi': '', 'pages': '', 'volume': '',
            })
        return out
    except Exception as e:
        return [{'error': f'{type(e).__name__}: {str(e)[:100]}'}]


def show(tag, results, source, note=''):
    print(f'\n{"="*78}\n[{tag}] ({source}) {note}')
    for i, r in enumerate(results, 1):
        if 'error' in r:
            print(f'  ! {r["error"]}')
            continue
        au = ', '.join(r.get('authors', []))[:105]
        v = r.get('venue') or ''
        extra = f"| cite:{r['cited']}" if r.get('cited') else ''
        print(f'  {i}. {r["title"][:92]}')
        print(f'     {au} | {r["year"]} | {v[:58]} | {r["type"]} {extra}')
        print(f'     doi:{r["doi"]} {r.get("volume","")} {r.get("pages","")} {r.get("arxiv_id","")}')


# group -> [(tag, source, query, note)]
GROUPS = {
    '1': [
        ('CITE-9 边缘计算', 'openalex', 'edge computing vision challenges', 'Shi et al. IoT-J 2016 类'),
        ('CITE-4 规则视频分析', 'openalex', 'rule-based video event detection surveillance framework', ''),
        ('CITE-16 智能零售柜商品识别', 'openalex', 'vending machine product recognition deep learning', ''),
        ('CITE-1 快递柜/外卖柜', 'openalex', 'parcel locker smart delivery last mile', ''),
        ('CITE-2 无人零售/零售损耗', 'openalex', 'unattended retail store security loss prevention', ''),
    ],
    '2': [
        ('CITE-16b 冰箱/柜内物品检测', 'openalex', 'refrigerator food item detection vision', ''),
        ('CITE-16c 货架商品检测', 'openalex', 'shelf product detection retail vision', ''),
        ('CITE-1b 校园配送', 'openalex', 'campus food delivery service', ''),
        ('CITE-15b 徘徊检测二次', 'openalex', 'abnormal behavior detection loitering trajectory', ''),
        ('CITE-11b 视频脱敏', 'openalex', 'video de-identification privacy surveillance', ''),
    ],
}

if __name__ == '__main__':
    grp = sys.argv[1] if len(sys.argv) > 1 else 'all'
    groups = GROUPS.keys() if grp == 'all' else [grp]
    for g in groups:
        for tag, src, q, note in GROUPS[g]:
            res = openalex(q) if src == 'openalex' else arxiv(q)
            show(tag, res, src, note)
            time.sleep(1.0)
