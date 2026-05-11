# gunicorn.conf.py — SafeAsset
workers      = 1
threads      = 2
timeout      = 120
keepalive    = 5
worker_class = 'sync'