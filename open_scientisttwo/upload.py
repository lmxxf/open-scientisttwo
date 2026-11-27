"""Official website client: URL request -> S3 multipart (file last) -> confirm."""
import json, uuid, urllib.request, urllib.error
from pathlib import Path


def multipart(fields, file=None):
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    if file:
        name, data = file
        if any(c in name for c in '\r\n"'):
            raise ValueError('unsafe filename')
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: application/pdf\r\n\r\n'.encode()+data+b'\r\n')
    return b''.join(parts)+f'--{boundary}--\r\n'.encode(), 'multipart/form-data; boundary='+boundary


class Client:
    def __init__(self, base='https://paperreview.ai', opener=urllib.request.urlopen):
        self.base, self.opener = base.rstrip('/'), opener

    def request(self, url, body=None, content_type=None, raw=False):
        req = urllib.request.Request(url, body, headers={'Content-Type': content_type} if content_type else {})
        try:
            with self.opener(req, timeout=60) as response:
                data = response.read()
                return response.status, data if raw else (json.loads(data) if data else {})
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RuntimeError('rate limited (429); no automatic resubmission') from exc
            raise

    def submit(self, pdf, venue='', email='', dry_run=True):
        pdf = Path(pdf)
        data = pdf.read_bytes()
        if not data.startswith(b'%PDF-'):
            raise ValueError('PDF required')
        if dry_run:
            return dict(dry_run=True, filename=pdf.name, bytes=len(data), steps=['get-upload-url','S3 POST file last','confirm-upload'])
        _, signed = self.request(self.base+'/api/get-upload-url', json.dumps(dict(filename=pdf.name, venue=venue)).encode(), 'application/json')
        if not signed.get('success') or not all(k in signed for k in ('presigned_url','s3_key','presigned_fields')):
            raise ValueError('invalid presign response')
        body, ct = multipart(signed['presigned_fields'], (pdf.name, data))
        s3_status, _ = self.request(signed['presigned_url'], body, ct, raw=True)
        if s3_status not in (200, 201, 204):
            raise ValueError('S3 upload failed')
        body, ct = multipart(dict(s3_key=signed['s3_key'], venue=venue, email=email))
        status, result = self.request(self.base+'/api/confirm-upload', body, ct)
        if status not in (200, 201, 202) or result.get('success') is not True or not isinstance(result.get('token'), str) or not result['token'].strip():
            raise ValueError('invalid confirmation success/token')
        return dict(status=status, token=result['token'], result=result)

    def review(self, token):
        from urllib.parse import quote
        status, result = self.request(self.base+'/api/review/'+quote(token, safe=''))
        return dict(status=status, pending=status == 202, result=result)
