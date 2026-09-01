# Notices

## Why this project is GPL-3.0-or-later

`wyoming-glados` links against [`piper-tts`](https://github.com/OHF-Voice/piper1-gpl),
which is **GPL-3.0-or-later**. The link is not incidental: `piper_runtime`
imports `piper.phonemize_espeak` to turn text into phonemes at request time,
and the published container image ships the package. Text has to become
phonemes before the VITS model can say anything, so the dependency is on the
critical path of every synthesis.

That makes the distributed program a combined work, and the GPL requires the
whole of it to be offered under GPL-3.0-or-later. The project was previously
MIT, which was incorrect for the artifact we publish.

## Prior MIT licence

Earlier versions of this project were released under the MIT licence. That
notice is preserved here as required; it is not removed by the relicensing, and
those earlier versions remain available under MIT.

```
MIT License

Copyright (c) 2023 Jonathan Simard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING OUT OF
OR IN CONNECTION WITH THE SOFTWARE, OR IN ANY WAY ARISING FROM, OR OUT OF, THE
SOFTWARE.
```

MIT is GPL-compatible, so the MIT-licensed portions may be distributed as part
of this GPL-3.0-or-later work.

## Voice model and training data

The voice is fine-tuned from the **LJSpeech** Piper checkpoint, which is public
domain. Lessac was deliberately avoided: the Blizzard 2013 corpus is licensed
for research only and forbids commercial use and redistribution, and fine-tuned
weights carry the base weights with them.

Transcripts and audio come from [theportalwiki.com](https://theportalwiki.com).
GLaDOS, Portal and Aperture Science are trademarks of Valve Corporation. This
project is unaffiliated with Valve, and the source recordings remain their
copyright; the corpus is redistributed for non-commercial fan use.
