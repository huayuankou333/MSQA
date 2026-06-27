# MMC-QA

Static GitHub Pages site for:

**MMC-QA: A Natively Sourced Multilingual and Multicultural Question Answering Benchmark**

This repository is maintained as a deployable static website. The files served by GitHub Pages are:

```text
index.html
assets/
```

The page is intended to be served from:

```text
https://huayuankou333.github.io/MMC-QA/
```

## Updating the Site

Build the page locally, then copy the generated static files into the repository root:

```bash
npm run build
```

After building, mirror the static output from `dist/` as:

```text
dist/index.html -> index.html
dist/assets/*   -> assets/
```

Commit and push those static files to `main`.

## GitHub Pages Settings

In the GitHub repository settings, use:

```text
Pages source: Deploy from a branch
Branch: main
Folder: / (root)
```

## Figure Assets

All figures are local PNG/JPG assets under `assets/`. Replace files with the same names to update figures without changing the page bundle.
