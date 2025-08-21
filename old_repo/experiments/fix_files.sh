# from the directory that currently contains those model folders
mkdir -p trained_models

# Preview what will happen (dry run)
for d in .._.._*; do
  [ -d "$d" ] || continue
  echo "$d  ->  trained_models/${d#.._.._}"
done

# If the preview looks right, do the move+rename
for d in .._.._*; do
  [ -d "$d" ] || continue
  mv -v -- "$d" "trained_models/${d#.._.._}"
done

# See results
ls -1 trained_models
