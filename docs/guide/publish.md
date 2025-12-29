
# Publishing

1. Create an index for L2 windows:![alt text](image.png)

```bash
kbctl index-l2 --digests-root ./digests/L2 --out-json ./index/l2_by_window.json
```

2. Copy to a publish directory:

```bash
kbctl publish --digests-root ./digests/L2 --out-dir ./digests/_published
```

You can serve `./digests/_published` with any static site or GitHub Pages.
