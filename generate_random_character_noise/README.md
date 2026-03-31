# generate_random_character_noise

Synthetic OCR noise generator with **script-specific confusable character
tables** for six writing systems:

| Script | Example confusables |
|--------|--------------------|
| `latin` | `öüäéèà ÜÄÖ` + ASCII |
| `cyrillic` | `абвгд…ёЁ` |
| `greek` | `αβγδ…άέήίόύώ` |
| `arabic` | `ابتث…ءآأؤإئ` |
| `hebrew` | `אבגד…ךםןףץ` |
| `georgian` | `აბგდ…ჰ` |

## Quick start

```bash
# Latin noise at CER ≈ 4 %
python generate_random_character_noise.py data.csv \
    --columns deu fra --script latin --cer 0.04 --seed 42 -o noised.csv

# Cyrillic noise
python generate_random_character_noise.py data.csv \
    --columns rus --script cyrillic --cer 0.04 -o noised.csv
```

Output adds `{col}_04` columns (configurable via `--suffix`).
