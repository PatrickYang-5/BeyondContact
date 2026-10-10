# Release validation

The original code-release checks below predate the supplied manuscript and raw `figs/` additions. Presentation checks for the updated README and `docs/` are recorded separately below.

- 73 existing task behavior checks passed (including fixed and widening goal-range cases).
- Airflow model numerical smoke test passed on CUDA.
- 65 Python files parsed successfully; local import targets resolve in this checkout.
- Exactly two training and two Play task registrations; all configuration entry points exist.
- XML mesh paths and all six USD layers' external references resolve within the release.
- Website local links resolve; the static site has no original repository links or external tracking scripts.
- Known original author/account, institution, repository and home-directory strings were absent from the released files and decoded USD layers.
- No symlinks, original Git history, experiment logs or pretrained checkpoints are included.

The existing goal-range test assumed an expanding curriculum although the released configuration uses a fixed range. In this copy, the test explicitly covers both configurations; task training parameters are unchanged.

Full Isaac Sim environment startup, rollout and training have not been validated for this isolated release. Static import checks and CPU task tests do not substitute for a simulator run. Numerical checks used an existing compatible Isaac Lab environment without installing this copy into that environment.

Third-party asset creator names remain as attribution. See THIRD_PARTY_NOTICES.md for asset provenance and the outstanding robot-asset license documentation gap. Anonymisation removes direct identifiers checked here; code similarity to other public work can still reveal provenance.

## Research presentation validation

- Updated English README and website content checked against the supplied manuscript, with anonymous author labeling. The identifying manuscript PDF is not linked or served by the website.
- All local HTML asset paths, section anchors, and README file links resolve. Image dimensions and alternative text are present.
- Optimized WebP figures and extracted posters contain no EXIF, XMP, or ICC metadata. Original supplied media remain unchanged.
- Four H.264 web videos preserve full duration and playback speed, omit audio, and have MP4 faststart metadata. All four pass a full decode check.
- Firefox browser checks verify image loading, video source/poster/caption/download switching, decoding of the RR video, and clipboard copying after a user click.
- Desktop (1440 px), tablet (768 px), and narrow layouts (390 px and 320 px) checked for page overflow and navigation visibility, including expanded result and training details.
- Videos use explicit playback controls and `preload="none"`; figures load lazily except the hero. The website uses local fonts/assets and no external scripts.

No robot code or training configuration was changed by the presentation update. The simulator validation limits above still apply.
