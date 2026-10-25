# Real-photo ground truth for the doses path

These transcripts were read off the PIXELS by a person looking at each photograph.
That is the whole point: the rest of this repo's accuracy numbers are measured
against strings its own renderer drew, which proves the model learned the renderer,
not the world.

Rules for adding a file here:
- one `.txt` per photograph, same basename as `../raw/<name>.<ext>`;
- line by line, as printed, top to bottom;
- `[?]` for a token that could not be read with certainty — never guess. A guessed
  label becomes a metric ceiling nobody can see;
- a photo with no legible panel does NOT get a file (it is not a page);
- do not "fix" spelling: a label that prints KAL. IODID. is ground truth spelled
  exactly that way.

Status: 2 photographs transcribed so far. Wikimedia's "medicine bottle" pool is
mostly ANTIQUE apothecary bottles with decorative, illegible script, or parody
labels — genuinely usable modern dose panels are rare there, so this set grows
mainly from photographs the product's own users take.
