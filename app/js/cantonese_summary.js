/** Deterministic HK Traditional Chinese summaries. No free-text translation,
 * medicine guessing, decimal-stripping normalisation, or cross-medicine doses.
 */
const CantoneseSummary = (() => {
  const digit = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
  function integer(n) {
    if (!Number.isSafeInteger(n) || n < 0 || n > 99999999) return null;
    if (n === 0) return "零";
    if (n >= 10000) {
      const high = integer(Math.floor(n / 10000)),
        low = n % 10000;
      return high + "萬" + (low ? (low < 1000 ? "零" : "") + integer(low) : "");
    }
    let out = "",
      zero = false;
    for (const [v, u] of [
      [1000, "千"],
      [100, "百"],
      [10, "十"],
      [1, ""],
    ]) {
      const d = Math.floor(n / v) % 10;
      if (d) {
        if (zero) out += "零";
        out += (v === 10 && d === 1 && !out ? "" : digit[d]) + u;
        zero = false;
      } else if (out && n % v) zero = true;
    }
    return out;
  }
  function number(token) {
    if (!/^\d{1,8}(?:\.\d+)?$/.test(token)) return null;
    const [a, b] = token.split(".");
    const i = integer(Number(a));
    return i === null
      ? null
      : i + (b ? "點" + [...b].map((c) => digit[Number(c)]).join("") : "");
  }
  function amount(raw) {
    const original = String(raw).trim();
    if (
      !/^(?:HK\$|HKD|USD|US\$|GBP|£|EUR|€|MYR|RM|\$)?\s*(?:\d{1,8}|\d{1,3}(?:,\d{3})+)(?:\.\d{2})?$/.test(
        original,
      )
    )
      return null;
    const s = original.replace(/,/g, "");
    const m =
      /^(HK\$|HKD|USD|US\$|GBP|£|EUR|€|MYR|RM|\$)?\s*(\d{1,8})(?:\.(\d{2}))?$/.exec(
        s,
      );
    if (!m) return null;
    const [, unit, a, b = "00"] = m;
    const n = integer(Number(a));
    if (unit === "HK$" || unit === "HKD")
      return (
        n +
        "蚊" +
        (b[0] !== "0" ? digit[+b[0]] + "毫" : "") +
        (b[1] !== "0" ? digit[+b[1]] + "仙" : "")
      );
    const units = {
      USD: "美元",
      US$: "美元",
      GBP: "英鎊",
      "£": "英鎊",
      EUR: "歐元",
      "€": "歐元",
      MYR: "馬來西亞令吉",
      RM: "馬來西亞令吉",
    };
    return number(a + "." + b) + (units[unit] || "（貨幣未確認）");
  }
  const receiptLabels = {
    Shop: "店舖",
    Date: "日期",
    Time: "時間",
    Total: "總金額",
    Subtotal: "小計",
    "Before tax": "稅前金額",
    Discount: "折扣",
    Tax: "稅項",
    "Payment method": "付款方式",
    "Card ending": "卡號末四位",
    "Receipt number": "收據編號原文",
    "Transaction number": "交易編號原文",
    Quantity: "數量",
  };
  const payment = {
    cash: "現金",
    visa: "Visa",
    mastercard: "Mastercard",
    contactless: "非接觸式付款",
  };
  const month = {
    jan: 1, january: 1, feb: 2, february: 2, mar: 3, march: 3,
    apr: 4, april: 4, may: 5, jun: 6, june: 6, jul: 7, july: 7,
    aug: 8, august: 8, sep: 9, sept: 9, september: 9, oct: 10,
    october: 10, nov: 11, november: 11, dec: 12, december: 12,
  };
  function spokenDate(raw) {
    const s = String(raw).trim();
    let m = /^(\d{4})-(\d{1,2})-(\d{1,2})$/.exec(s);
    if (!m) {
      const named = /^(\d{1,2})[- ]([A-Za-z]+)[, -]+(\d{4})$/.exec(s);
      if (named && month[named[2].toLowerCase()])
        m = [s, named[3], month[named[2].toLowerCase()], named[1]];
    }
    if (!m) return null;
    const y = number(m[1]), mo = number(String(Number(m[2]))), d = number(String(Number(m[3])));
    return y && mo && d ? y + "年" + mo + "月" + d + "日" : null;
  }
  function build(result) {
    if (!result) return { eligible: false, text: "請先掃描文件。" };
    const pageScope = result.pageNote ? "只包括目前辨識嘅文件頁面。" : "";
    if (result.category === "receipt") {
      const fields = (result.fields || []).filter(
        (f) =>
          Number.isFinite(f.confidence) && f.confidence >= 0.75 && f.sourceLine,
      );
      const lines = [];
      for (const f of fields) {
        const label = receiptLabels[f.label];
        if (!label) continue;
        let value = String(f.value);
        if (/^(Total|Subtotal|Before tax|Discount|Tax)$/.test(f.label)) {
          value = amount(value);
          if (!value) continue;
        }
        if (
          f.label === "Shop" &&
          /^PARKN'?SHOP$/i.test(value.replace(/\s/g, ""))
        )
          value = "百佳超級市場（PARKnSHOP）";
        if (f.label === "Payment method")
          value = payment[value.toLowerCase()] || value;
        if (f.label === "Date") value = spokenDate(value) || value;
        if (f.label === "Quantity") {
          const quantity = /^(\d+(?:\.\d+)?)\s*(?:pcs?|pieces?|items?|ea|each)$/i.exec(value.trim());
          if (quantity) value = number(quantity[1]) + "件";
        }
        lines.push(label + "係" + value + "。");
      }
      return {
        eligible: lines.length > 0,
        text: lines.length
          ? pageScope +
            "呢張係收據。" +
            lines.join("") +
            "未能確認嘅資料唔會讀出，請核對收據原文。"
          : "收據資料未夠清楚，請再影一張。",
        partial: true,
      };
    }
    if (result.category !== "medical")
      return {
        eligible: false,
        text: "呢種文件未支援廣東話摘要，請核對原文，或轉用英文。",
      };
    if (!result.speechText)
      return {
        eligible: false,
        text: "資料未夠清楚，唔會亂估。請核對原文，或請藥劑師幫忙。",
      };
    const text = String(result.speechText),
      literal = (s) => String(s).toLowerCase().replace(/\s/g, "");
    const names = [
      ...new Set(
        (result.entities || [])
          .filter(
            (e) =>
              e.label === "DRUG" &&
              e.confidence >= 0.75 &&
              literal(text).includes(literal(e.text)),
          )
          .map((e) => e.text),
      ),
    ];
    const doses = [
      ...new Set(
        (result.fields || [])
          .filter((f) => f.label === "Dosage" && f.confidence >= 0.75)
          .map((f) => f.value),
      ),
    ];
    if (names.length !== 1 || doses.length > 1)
      return {
        eligible: false,
        text: "未能安全配對藥名同用量。請揀英文聽原文，再同藥劑師確認。",
      };
    const lines = [
      "呢張係藥物標籤。原文藥名係" +
        names[0] +
        "。藥名保留原文，唔會估中文譯名。",
    ];
    if (doses.length) {
      const m = /^(\d{1,5}(?:\.\d+)?)\s*(mg|mcg|g|ml|iu)$/i.exec(doses[0]);
      if (m && literal(text).includes(literal(doses[0])))
        lines.push(
          "原文強度或用量係" +
            number(m[1]) +
            { mg: "毫克", mcg: "微克", g: "克", ml: "毫升", iu: "國際單位" }[
              m[2].toLowerCase()
            ] +
            "。",
        );
    }
    // Full-line allow-list: never match a short phrase inside a negation or a longer dose.
    const phrases = {
      "take one capsule three times daily": "每日三次，每次一粒膠囊。",
      "take one tablet three times daily": "每日三次，每次一粒藥片。",
      "take one capsule twice daily": "每日兩次，每次一粒膠囊。",
      "take one tablet twice daily": "每日兩次，每次一粒藥片。",
      "take one tablet once daily": "每日一次，每次一粒藥片。",
      "take one capsule once daily": "每日一次，每次一粒膠囊。",
      "keep out of reach of children": "請放喺兒童接觸唔到嘅地方。",
      "may cause drowsiness": "可能引起睡意。",
      "do not exceed the stated dose": "唔好超過標籤指定嘅用量。",
      "for external use only": "只供外用。",
    };
    for (const line of text.split("\n")) {
      const key = line.trim().toLowerCase().replace(/[.!]$/, "");
      if (phrases[key]) lines.push(phrases[key]);
    }
    lines.push(
      "只讀出已核對同支援嘅資料。其餘指示同警告可能未有翻譯，請核對原文，並同藥劑師確認。",
    );
    return { eligible: true, text: pageScope + lines.join(""), partial: true };
  }
  return { build, number, amount };
})();
if (typeof window !== "undefined") window.CantoneseSummary = CantoneseSummary;
if (typeof module !== "undefined" && module.exports)
  module.exports = CantoneseSummary;
