# Licences

Everything the page ships or fetches, and the toolchain behind it. No model weights or libraries
come from anywhere else: the model is trained in your browser, and the recorded run was trained
by this code.

## Shipped with the page

| what | version | licence | size | source |
| --- | --- | --- | --- | --- |
| This project's code | 0.1.0 | MIT (`LICENSE`) | about 120 KB of JavaScript, gzipped | this repository |
| Recorded run, `public/recorded/run.json`, `features.json` | 2 Nov 2026 | MIT, as part of this project | 94 KB and 223 KB | trained by this code, `npm run record` |
| Recorded weights, `public/recorded/weights.bin` (model 817,920 and autoencoder 263,296 half-precision values) | 2 Nov 2026 | MIT, as part of this project | 2,162,432 bytes | trained by this code, `npm run record`; fetched only when asked for; not committed (model weights stay out of the repository) |
| B612 and B612 Mono | Google Fonts release | SIL Open Font Licence 1.1 (`public/fonts/OFL.txt`) | 60 KB | https://github.com/polarsys/b612, downloaded once from Google Fonts |
| Shakespeare's Sonnets, Alice's Adventures in Wonderland, Hamlet, Macbeth, Romeo and Juliet, Othello, King Lear | fetched 1 Nov 2026 | public domain in the United States (Project Gutenberg) | 1.0 MB of text | https://www.gutenberg.org, ebooks 1041, 11, 1524, 1533, 1513, 1531, 1532 |
| Webster's Second New International Dictionary word list | BSD Unix `web2` | copyright lapsed (per its README) | 320 KB | `/usr/share/dict/web2` |

The page has no runtime dependencies: no machine-learning library, no framework.

## Toolchain (development only)

`npm run licences` checks every package in `package-lock.json` against MIT, Apache-2.0, BSD and
ISC. Last run: 101 packages (MIT 83, MIT or Apache-2.0 11, Apache-2.0 4, BSD-3-Clause 2, ISC 1),
all on the list. The browser tests use Playwright 1.63 (Apache-2.0) and its Chromium, WebKit and
Firefox builds.
