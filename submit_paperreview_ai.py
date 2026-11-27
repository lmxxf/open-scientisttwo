"""Official upload/query CLI; polling never resubmits."""
import argparse, json, time
from pathlib import Path
from open_scientisttwo.upload import Client

def main():
    p = argparse.ArgumentParser()
    p.add_argument('pdf', nargs='?')
    p.add_argument('--token')
    p.add_argument('--venue', default='')
    p.add_argument('--email', default='')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--poll', type=int, default=0, help='maximum additional status queries (0..20)')
    p.add_argument('--interval', type=float, default=5)
    a = p.parse_args()
    if not 0 <= a.poll <= 20 or not 0 <= a.interval <= 60:
        p.error('poll 0..20, interval 0..60 required')
    if bool(a.pdf) == bool(a.token):
        p.error('provide PDF or --token')
    client = Client()
    receipt = client.review(a.token) if a.token else client.submit(a.pdf, a.venue, a.email, a.dry_run)
    token = a.token or receipt.get('token')
    def persist():
        if a.output:
            a.output.write_text(json.dumps(receipt, indent=2))
    persist()  # save token before any optional network query
    for _ in range(a.poll):
        if not token or receipt.get('pending') is False:
            break
        time.sleep(a.interval)
        receipt['query'] = client.review(token)
        persist()
        if not receipt['query']['pending']:
            break
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
