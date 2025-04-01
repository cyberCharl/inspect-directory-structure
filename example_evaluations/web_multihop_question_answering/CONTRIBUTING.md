# Contributing Guide

## Maintenance Details

This evaluation should be fairly low maintenance, but involves some aspects that may change or break over time. Notably, it connects to the web with inspect's web browsing tool using inspect's web browsing docker image. Both the specific web pages that the model may access, and inspect's docker image may change over time, and could cause failures down the line.

It's unfortunately not currently possible to pin the version of the web_browsing docker image (it's tied to the inspect version), so this could cause breakages / changes in behaviour if/when inspect is upgraded.

## Testing Details

Run tests with pytest from the root of the evaluation folder:

```bash
pytest
```

## Developer Contact Details

Sid Black: `sid.black@dsit.gov.uk`
