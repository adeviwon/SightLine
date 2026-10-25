/** Consent belongs to the caller. IndexedDB transactions are the sole Saved authority.
 * No localStorage documents, no upload, no storage writes merely by scanning.
 */
const SightLineHistory = (() => {
  const NAME = "sightline-scans",
    VERSION = 1;
  function open() {
    return new Promise((resolve, reject) => {
      if (typeof indexedDB === "undefined") {
        reject(new Error("History storage is unavailable in this browser."));
        return;
      }
      let req,
        blocked = false;
      try {
        req = indexedDB.open(NAME, VERSION);
      } catch (e) {
        reject(e);
        return;
      }
      req.onupgradeneeded = () => {
        const db = req.result;
        if (blocked) {
          db.close();
          return;
        }
        if (!db.objectStoreNames.contains("scans"))
          db.createObjectStore("scans", { keyPath: "id" });
      };
      req.onerror = () => reject(req.error);
      req.onblocked = () => {
        blocked = true;
        reject(new Error("Close other SightLine tabs, then try again."));
      };
      req.onsuccess = () => {
        const db = req.result;
        if (blocked) {
          db.close();
          return;
        }
        db.onversionchange = () => db.close();
        resolve(db);
      };
    });
  }
  async function operation(mode, action) {
    const db = await open();
    return new Promise((resolve, reject) => {
      let value;
      const tx = db.transaction("scans", mode);
      tx.oncomplete = () => {
        db.close();
        resolve(value);
      };
      tx.onabort = tx.onerror = () => {
        db.close();
        reject(tx.error || new Error("The scan was not saved."));
      };
      try {
        const request = action(tx.objectStore("scans"));
        request.onsuccess = () => {
          value = request.result;
        };
      } catch (e) {
        tx.abort();
        reject(e);
      }
    });
  }
  function snapshot(result, image = null, id) {
    if (
      !result ||
      typeof result.ocrText !== "string" ||
      !Array.isArray(result.fields)
    )
      throw new Error("There is no scan to save.");
    return {
      id: id || crypto.randomUUID(),
      schema: 1,
      savedAt: new Date().toISOString(),
      result: JSON.parse(JSON.stringify(result)),
      image: image instanceof Blob ? image : null,
    };
  }
  const save = (result, image, id) =>
    operation("readwrite", (s) => s.put(snapshot(result, image, id)));
  async function list() {
    return (await operation("readonly", (s) => s.getAll())).sort((a, b) =>
      (b.result?.timestamp || b.savedAt).localeCompare(
        a.result?.timestamp || a.savedAt,
      ),
    );
  }
  async function get(id) {
    const r = await operation("readonly", (s) => s.get(id));
    if (r && r.schema !== 1)
      throw new Error("This saved scan cannot be opened by this version.");
    return r;
  }
  function message(error) {
    return error?.name === "QuotaExceededError"
      ? "Storage is full. Delete older saved scans, or save without an image."
      : error?.name === "SecurityError" || error?.name === "InvalidStateError"
        ? "History storage is blocked. Check browser privacy settings, or try the installed app."
        : error?.message ||
          "History could not be saved. Your current scan is still available.";
  }
  return {
    save,
    list,
    get,
    remove: (id) => operation("readwrite", (s) => s.delete(id)),
    clear: () => operation("readwrite", (s) => s.clear()),
    message,
  };
})();
if (typeof window !== "undefined") window.SightLineHistory = SightLineHistory;
if (typeof module !== "undefined" && module.exports)
  module.exports = SightLineHistory;
