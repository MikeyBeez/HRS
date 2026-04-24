#!/bin/bash
cd ~/Code/HRS  # adjust path if different

OUT=/tmp/hrs_review.txt
> $OUT

echo "=== DIRECTORY STRUCTURE ===" >> $OUT
echo "" >> $OUT
echo "--- experiments/identity_ae/ ---" >> $OUT
ls experiments/identity_ae/ 2>/dev/null | sort >> $OUT
echo "" >> $OUT
echo "--- results/identity_ae/ ---" >> $OUT
ls results/identity_ae/ 2>/dev/null | sort >> $OUT
echo "" >> $OUT

echo "=== README ===" >> $OUT
echo "" >> $OUT
cat README.md 2>/dev/null >> $OUT
echo "" >> $OUT

echo "=== DOCS (if any) ===" >> $OUT
echo "" >> $OUT
for f in docs/*.md *.md; do
  if [ -f "$f" ] && [ "$f" != "README.md" ]; then
    echo "--- $f ---" >> $OUT
    cat "$f" >> $OUT
    echo "" >> $OUT
  fi
done

echo "=== KEY RESULT JSONs ===" >> $OUT
echo "" >> $OUT
for phase in 32 33 35 47 52 55 57 58 59 60 61 62 63 64 65 66 67 68; do
  for jsonfile in results/identity_ae/phase${phase}/*.json results/identity_ae/phase${phase}*/*.json; do
    if [ -f "$jsonfile" ]; then
      echo "--- $jsonfile ---" >> $OUT
      cat "$jsonfile" >> $OUT
      echo "" >> $OUT
    fi
  done
done

echo "=== EXPERIMENT SCRIPT HEADERS ===" >> $OUT
echo "(first 30 lines of each phase script for context)" >> $OUT
echo "" >> $OUT
for script in experiments/identity_ae/phase*.py; do
  if [ -f "$script" ]; then
    echo "--- $script ---" >> $OUT
    head -30 "$script" >> $OUT
    echo "" >> $OUT
  fi
done

echo "" >> $OUT
echo "=== END ===" >> $OUT
echo "Output written to $OUT"
wc -l $OUT
