/** Small closed-set presentation layer for English OCR fields. The result passed
 * in is never modified: untranslated values and all confidence remain verbatim.
 */
const SightLineFieldLocalization = (() => {
  const labels = {
    receipt: {
      Shop: "商店",
      Address: "地址",
      Phone: "電話",
      "Kind of shop": "商店類別",
      Date: "日期",
      Time: "時間",
      Total: "總金額",
      Subtotal: "小計",
      "Before tax": "稅前小計",
      Discount: "折扣",
      Tax: "稅項",
      "Payment method": "付款方式",
      "Card ending": "卡號末四位",
      "Receipt number": "收據編號",
      "Transaction number": "交易編號",
      Quantity: "數量",
    },
    medical: {
      "Medicine type": "藥物劑型",
      "Strength on the label": "標籤上的藥物強度",
      "Amount each time": "每次用量",
      "When to take it": "何時服用",
      "How long": "服用多久",
      Medicine: "藥物",
      Medication: "藥名（保留英文原文）",
      Dosage: "強度或用量",
      Frequency: "服用次數",
      Quantity: "數量",
      Instructions: "服用指示",
      Warning: "警告",
      Warnings: "警告",
      Prescriber: "處方者",
      Doctor: "醫生",
      Patient: "病人",
      "Strength or dose text": "強度或用量",
      "Medicine name": "藥名（保留英文原文）",
    },
  };
  const payments = new Map([
    ["cash", "現金"],
    ["credit card", "信用卡"],
    ["debit card", "扣帳卡"],
    ["contactless", "非接觸式付款"],
  ]);
  const shops = new Map([
    ["supermarket", "超級市場"],
    ["pharmacy", "藥房"],
    ["restaurant", "餐廳"],
    ["cafe", "咖啡店"],
  ]);
  const medicalPhrases = new Map([
    ["take one capsule three times daily", "每日三次，每次一粒膠囊"],
    ["take one tablet three times daily", "每日三次，每次一粒藥片"],
    ["take one capsule twice daily", "每日兩次，每次一粒膠囊"],
    ["take one tablet twice daily", "每日兩次，每次一粒藥片"],
    ["take one tablet once daily", "每日一次，每次一粒藥片"],
    ["take one capsule once daily", "每日一次，每次一粒膠囊"],
    ["once daily", "每日一次"],
    ["twice daily", "每日兩次"],
    ["three times daily", "每日三次"],
    ["four times daily", "每日四次"],
    ["keep out of reach of children", "請放喺兒童接觸唔到嘅地方"],
    ["may cause drowsiness", "可能引起睡意"],
    ["do not exceed the stated dose", "唔好超過標籤指定嘅用量"],
    ["for external use only", "只供外用"],
  ]);
  const units = new Map([
    ["mg", "毫克"],
    ["mcg", "微克"],
    ["ug", "微克"],
    ["µg", "微克"],
    ["g", "克"],
    ["gm", "克"],
    ["ml", "毫升"],
    ["l", "公升"],
    ["iu", "國際單位"],
    ["i.u.", "國際單位"],
    ["tablet", "藥片"],
    ["tab", "藥片"],
    ["capsule", "膠囊"],
    ["cap", "膠囊"],
    ["drop", "滴"],
    ["puff", "噴"],
    ["spray", "噴劑"],
    ["patch", "貼片"],
    ["unit", "單位"],
  ]);
  function sourceContains(result, field) {
    const raw = String(result?.ocrText || result?.normalized || "");
    const val = String(field?.value || "").trim();
    if (!val) return false;
    const numeric = /^\D*\d/.test(val);
    const sourceLines = raw.split(/\r?\n/);
    return sourceLines.some((line) => {
      let from = 0;
      for (;;) {
        const at = line.toLocaleLowerCase().indexOf(val.toLocaleLowerCase(), from);
        if (at < 0) return false;
        const before = line[at - 1] || "";
        const after = line[at + val.length] || "";
        const adjacent = numeric ? /[\d.,]/ : /[a-z\d]/i;
        if (!adjacent.test(before) && !adjacent.test(after)) return true;
        from = at + 1;
      }
    });
  }
  function dateValue(raw) {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(raw.trim());
    return m ? `${m[1]}年${m[2]}月${m[3]}日` : null;
  }
  function medicalValue(field, raw) {
    const norm = raw.trim().toLowerCase().replace(/[.!]$/, "");
    if (medicalPhrases.has(norm)) return medicalPhrases.get(norm);
    const forms = { tablet:'藥片', tablets:'藥片', capsule:'膠囊', capsules:'膠囊', drops:'滴劑', patches:'貼片', inhaler:'吸入劑', inhalers:'吸入劑' };
    if (field.label === 'Medicine type' && forms[norm]) return forms[norm];
    const countWords = {one:'一',two:'兩',three:'三',four:'四',half:'半'};
    const count = /^(one|two|three|four|half)\s+(tablets?|capsules?)$/i.exec(norm);
    if (count) return countWords[count[1]] + '粒' + (/tablet/.test(count[2]) ? '藥片' : '膠囊');
    const duration = /^for\s+(\d+)\s+(days?|weeks?|months?)$/i.exec(norm);
    if (duration) return '持續 ' + duration[1] + ' ' + (/day/.test(duration[2]) ? '日' : /week/.test(duration[2]) ? '星期' : '個月');
    const m = /^(\d+(?:\.\d+)?)(\s*)(mg|mcg|ug|µg|g|gm|ml|l|iu|i\.u\.|tablets?|tabs?|capsules?|caps?|drops?|puffs?|sprays?|patches?|units?)$/i.exec(raw.trim());
    if (m) {
      const u = m[3].toLowerCase().replace(/s$/, "");
      const translatedUnit = units.get(u);
      if (translatedUnit) return m[1] + " " + translatedUnit;
    }
    // Medicine/person names and any unreviewed medical wording stay in English.
    if (/^(?:Medication|Medicine|Medicine name|Patient|Prescriber)$/.test(field.label))
      return { value: raw, retained: true };
    return null;
  }
  function medicalName(field) {
    return /^(?:Medication|Medicine|Medicine name|Patient|Prescriber)$/.test(field.label);
  }
  function receiptValue(field, raw) {
    if (
      field.label === "Shop" &&
      /^PARKN'?SHOP$/i.test(raw.replace(/\s/g, ""))
    )
      return "百佳超級市場（PARKnSHOP）";
    if (field.label === "Payment method")
      return payments.get(raw.trim().toLowerCase()) || null;
    if (field.label === "Kind of shop")
      return shops.get(raw.trim().toLowerCase()) || null;
    if (field.label === "Date") return dateValue(raw);
    if (field.label === "Quantity") {
      const m = /^(\d+(?:\.\d+)?)\s*(?:pcs?|pieces?|items?|ea|each)$/i.exec(raw.trim());
      if (m) return m[1] + " 件";
    }
    // Prices, dates outside an unambiguous ISO form, account/reference numbers,
    // tax and unknown merchant names retain their exact OCR spelling.
    return null;
  }
  function localize(result, field, language = "zh-HK") {
    const category = result?.category === "receipt" ? "receipt" : result?.category === "medical" ? "medical" : "other";
    const label = String(field?.label || "");
    const raw = String(field?.value ?? "");
    if (language !== "zh-HK")
      return { label, value: raw, lang: "en", translated: false, retained: true };
    const localizedLabel = (labels[category] || {})[label] || null;
    const confidenceTrusted =
      Number.isFinite(field?.confidence) && field.confidence >= 0.75;
    const literalSource = sourceContains(result, field);
    const candidate = literalSource && category === "medical" && medicalName(field)
      ? { value: raw, retained: true }
      : confidenceTrusted && literalSource
        ? category === "receipt"
          ? receiptValue(field, raw)
          : medicalValue(field, raw)
        : null;
    const value = candidate && typeof candidate === "object" ? candidate.value : candidate;
    const retained = !!candidate && typeof candidate === "object" && candidate.retained;
    return {
      label: localizedLabel || label,
      value: value || raw,
      lang: value && !retained ? "zh-HK" : "en",
      translated: !!value && !retained && value !== raw,
      retained: !!retained,
      needsVerification:
        !localizedLabel ||
        (!value &&
          !retained &&
          !(category === "receipt" && ["Date", "Time", "Total", "Subtotal", "Before tax", "Discount", "Tax", "Card ending", "Receipt number", "Transaction number"].includes(label))),
    };
  }
  return { localize, labels, medicalPhrases, units };
})();
if (typeof window !== "undefined") window.SightLineFieldLocalization = SightLineFieldLocalization;
if (typeof module !== "undefined" && module.exports) module.exports = SightLineFieldLocalization;
