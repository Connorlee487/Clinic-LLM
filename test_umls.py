import requests
API_KEY = os.environ.get("UMLS_API_KEY")
r = requests.get(
    'https://uts-ws.nlm.nih.gov/rest/search/current',
    params={'string': 'hypertension', 'apiKey': API_KEY, 'pageSize': 1}
)
results = r.json().get('result', {}).get('results', [])
print('API works!' if results else 'API failed')
print('First result:', results[0]['name'] if results else 'None')
