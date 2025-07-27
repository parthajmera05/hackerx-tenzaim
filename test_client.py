

import requests

headers = {
    "Authorization": "Bearer supersecurekey123",
    "Content-Type": "application/json"
}

data = {
    "documents": [
        "E:\hack\BAJHLIP23020V012223.pdf",
        "E:\hack\CHOTGDP23004V012223.pdf",
        "E:\hack\EDLHLGA23009V012223.pdf",
        "E:\hack\HDFHLIP23024V072223.pdf",
        "E:\hack\ICIHLIP22012V012223.pdf"
    ],
    "questions": [
        
        "Is maternity covered?"
    ]
}

res = requests.post("http://127.0.0.1:8000/hackrx/run", json=data, headers=headers)
print(res.status_code)
print(res.json())
