/** Interface-only translations. OCR text, field values and saved titles are excluded. */
const SightLineI18n = (() => {
  let lang = "en-GB";
  const sources = new WeakMap();
  const zh = {
    Medicine: "藥物",
    Shop: "店舖",
    Purchase: "購買資料",
    Totals: "金額",
    Payment: "付款",
    "Other details": "其他資料",
    Prescriber: "處方者",
    Warnings: "警告",
    "Personal details": "個人資料",
    "Skip to the reader": "跳到閱讀區",
    "SightLine home": "SightLine 主頁",
    "Reader tools": "閱讀工具",
    "Reading preferences": "閱讀設定",
    "Help using SightLine": "SightLine 使用說明",
    "Checking offline access": "正在檢查離線功能",
    "Ready offline": "可離線使用",
    "Medicine type": "藥物劑型",
    "Strength on the label": "標籤上的藥物強度",
    "Amount shown on the label": "標籤上辨認到的用量",
    "How much at once": "每次用量",
    "Amount each time": "每次用量",
    "When to take it": "何時服用",
    "How long": "服用多久",
    "Not clearly found:": "未能清楚辨認：",
    "Check the original label for missing details.": "請核對原標籤上尚未辨認的資料。",
    "Reader needs reloading": "請重新載入閱讀器",
    "The reading software could not load.": "未能載入閱讀軟件。",
    "Reload reader": "重新載入閱讀器",
    "Reload the reader, then choose your photo again. Saved scans will stay on this device.": "請重新載入閱讀器，然後再選擇相片。已儲存的掃描會保留在此裝置。",
    "Details for a helper": "協助者參考資料",
    "Let’s try that again.": "請再試一次。",
    "Let's try that again.": "請再試一次。",
    "Back to the reader": "返回閱讀器",
    "A clearer photo or another file may help.": "可以換一張較清晰的相片，或選擇另一份文件。",
    "Preparing offline reading": "正在準備離線閱讀",
    "Checking offline reading": "正在檢查離線閱讀",
    "Saving for offline reading": "正在儲存離線檔案",
    "Reading on this device": "正在本機辨識",
    "Private on this device": "只在本機處理",
    "Offline saving is unavailable": "此瀏覽器未能儲存離線檔案",
    "Offline saving did not finish": "離線檔案未儲存完成",
    "Offline. Reading files may be missing.": "離線中，閱讀檔案可能未齊備。",
    "A little clarity. A little more independence.": "睇得清楚啲，生活自主啲。",
    "Make medicine labels easier to read.": "輕鬆睇清藥物標籤",
    "Scan with Camera": "用相機掃描",
    "Upload Photo or PDF": "上載相片或 PDF",
    Back: "返回",
    "Scan another": "掃描另一份文件",
    Cancel: "取消",
    Retake: "重影",
    "Take photo": "影相",
    "Read this photo": "辨識這張相片",
    "Is every word in view?": "全部文字都入鏡嗎？",
    "Reading your document…": "正在辨識文件⋯",
    "What we read.": "辨識結果",
    "Read Aloud": "朗讀",
    "Hear reading help": "聽取閱讀協助",
    Stop: "停止",
    "Stop reading": "停止朗讀",
    "Check the original. Ask a pharmacist if unsure.":
      "請核對原文。如有疑問，請向藥劑師查詢。",
    "Check important amounts against the original receipt.":
      "重要金額請核對收據原文。",
    "Some details are unclear. They won’t be read aloud.":
      "部分資料未夠清楚，唔會讀出。",
    Receipt: "收據",
    "Medicine label or prescription": "藥物標籤或處方",
    Document: "文件",
    "All recognised text": "全部辨識原文",
    "View original image": "查看原圖",
    "Details for a helper": "供協助者參考",
    "Make it easier to read": "閱讀選項",
    "Reading language: English": "閱讀語言：英文",
    "Reading language: Cantonese": "閱讀語言：廣東話",
    "Listen to the clear receipt details. Unclear amounts are left out.":
      "可以聽清楚辨識到嘅收據資料，未確認金額唔會讀出。",
    "Listen at your own pace. You can stop the reading at any time.":
      "按自己步伐聆聽，隨時可以停止。",
    "A little clarity, every day.": "每天，睇得清楚啲。",
    "Help & privacy": "說明及私隱",
    "Your reading preferences": "閱讀設定",
    "Close preferences": "關閉設定",
    "Choose what feels comfortable. Changes save on this device.":
      "揀選適合自己嘅設定，會記喺呢部裝置。",
    Appearance: "顯示模式",
    Light: "淺色",
    Dark: "深色",
    "Match my device": "跟隨裝置",
    "Reading language": "閱讀語言",
    "More reading options": "更多閱讀選項",
    "Reading method": "辨識方式",
    "Enhanced reader": "增強辨識",
    "Original reader": "原有辨識",
    "Read without extra image cleanup": "不使用額外影像處理",
    "Use simple document matching": "使用簡單文件分類",
    Done: "完成",
    "A clearer read": "睇得更清楚",
    "Close help": "關閉說明",
    "For a good photo": "影相小貼士",
    "Your privacy": "你的私隱",
    "Reading offline": "離線閱讀",
    "Check important information": "核對重要資料",
    "About SightLine": "關於 SightLine",
    "Model credits": "模型來源",
    "Check this device": "檢查裝置",
    "Check Device": "檢查裝置",
    "Got it": "知道了",
    "Device check": "裝置檢查",
    "← Back to SightLine": "← 返回 SightLine",
    "Return to SightLine": "返回 SightLine",
    "Try saving for offline use": "重新儲存離線檔案",
    Purchase: "購物資料",
    Totals: "金額",
    Payment: "付款",
    Reference: "參考資料",
    Medicine: "藥物",
    "Strength and directions": "藥物強度及用法",
    Warnings: "警告",
    "Personal details": "個人資料",
    Prescriber: "處方者",
    "Other details": "其他資料",
    Shop: "店舖",
    "Kind of shop": "商店類別",
    Date: "日期",
    Time: "時間",
    Total: "總金額",
    "Before tax": "稅前金額",
    Subtotal: "小計",
    Discount: "折扣",
    Tax: "稅項",
    "Payment method": "付款方式",
    Address: "地址",
    Phone: "電話",
    Instructions: "服用指示",
    "English text — no verified translation": "英文原文，暫未有核實譯文",
    "English original: ": "英文原文：",
    "Medicine name": "原文藥名",
    "Strength or dose text": "原文強度或用量",
    Frequency: "服用次數",
    Warning: "警告",
    Patient: "病人",
    "Receipt number": "收據編號",
    "Transaction number": "交易編號",
    "Card ending": "卡號末四位",
    Medication: "藥物",
    Dosage: "強度或用量",
    History: "紀錄",
    "Saved scans": "已儲存文件",
    "Your saved scans": "你的掃描紀錄",
    "No saved scans yet": "未有儲存紀錄",
    "Open scan": "開啟文件",
    Delete: "刪除",
    "Delete all": "全部刪除",
    "Save scan": "儲存文件",
    "Saved on this device": "已儲存在此裝置",
    "Saving…": "正在儲存⋯",
    "Save scans on this device": "在此裝置儲存掃描紀錄",
    "Also save original images": "同時儲存原圖",
    "Save this scan": "儲存這份文件",
    "Keep scan history?": "要儲存掃描紀錄嗎？",
    "Enable saving": "啟用儲存",
    "Not now": "暫時不要",
    "Delete saved scan?": "刪除已儲存文件？",
    "Delete all saved scans?": "刪除全部掃描紀錄？",
    "Keep scans": "保留紀錄",
    "Confirm deletion": "確認刪除",
    Language: "語言",
    "Use English": "轉用英文",
    "New scans are saved only after you enable this preference.":
      "啟用呢項設定後，先會自動儲存新文件。",
    "Images are optional. Scans stay on this device and are never uploaded.":
      "原圖可選擇唔儲存。文件只留喺呢部裝置，唔會上載。",
    "Anyone using SightLine on this device can open saved scans. Clearing browser data or reinstalling can erase them.":
      "使用呢部裝置嘅 SightLine 嘅人可以開啟紀錄。清除瀏覽資料或重裝程式可能會刪走紀錄。",
    "Turning saving off does not delete existing scans. Delete them in History.":
      "關閉儲存唔會刪除現有紀錄，請到掃描紀錄刪除。",
    "Earlier scans were not stored and cannot be recovered automatically.":
      "之前未有儲存嘅文件，無法自動搵返。",
    "Save a scan from its results, or enable saving in preferences.":
      "可以喺結果頁儲存文件，或到設定啟用自動儲存。",
    "Saving is off. Nothing is stored automatically.":
      "自動儲存已關閉，唔會自動保留文件。",
    "Original image was not saved. Recognised text is available below.":
      "呢份文件未有儲存原圖，可以睇下面嘅辨識原文。",
    "All saved scans will be deleted. Automatic saving will turn off.":
      "將刪除全部紀錄，並關閉自動儲存。",
    "This saved copy will be deleted from this device.":
      "將從呢部裝置刪除呢份已儲存文件。",
    "Language changes the interface and supported summaries. Original document text stays unchanged.":
      "語言選項只會改變介面同支援嘅摘要，文件原文保持不變。",
    "Summary to read aloud": "將會朗讀嘅摘要",
    "These clear details are prepared for reading aloud.":
      "以下清晰辨認到的資料會用作朗讀。",
    "These cards show the key fields. The full verified label text will also be read aloud.":
      "以下卡片顯示主要資料，亦會朗讀已核對的標籤原文。",
    "These cards organize key details. The reading follows its supported summary; check the original for anything missing.":
      "以下卡片整理主要資料，朗讀內容只包括支援的摘要。請核對原文有否其他資料。",
    "Preparing the voice…": "正在準備語音⋯",
    "No Cantonese voice is available. Install an offline Cantonese voice in your device speech settings, or choose English.":
      "未有可用嘅廣東話語音。請喺裝置語音設定下載離線廣東話語音，或揀英文。",
    "No offline English voice is available. Install an English voice in your device speech settings, then try again.":
      "未有可用嘅離線英文語音。請喺裝置語音設定下載英文語音，再試一次。",
    "Scans are kept only if you enable saving. Photos and text are never uploaded.":
      "只有啟用儲存，先會保留文件。相片同文字唔會上載。",
    "Scan a document before reading aloud.": "請先掃描文件。",
    "No receipt details are clear enough to read aloud. Try a clearer photo.":
      "收據資料未夠清楚，請再影一張。",
    "No clear words are available to read.": "未有清楚嘅文字可以朗讀。",
    "Medicine details or other words need checking. Try a clearer photo, or ask someone to check the document.":
      "藥物資料或其他文字需要核對。請再影一張，或請人幫忙核對。",
    "Finding an offline voice…": "正在尋找離線語音⋯",
    "Starting the voice…": "正在啟動語音⋯",
    "Reading aloud. Tap Stop reading to stop.":
      "正在朗讀，撳「停止朗讀」可以停止。",
    "Reading finished. You can listen again.": "朗讀完成，可以再聽一次。",
    "Reading stopped. Tap Read Aloud to listen again.":
      "已停止朗讀，可以撳「朗讀」再聽。",
    "Reading aloud on your iPhone speaker. Tap Stop reading to stop.":
      "正在使用 iPhone 喇叭朗讀。撳「停止朗讀」即可停止。",
    "Reading aloud through connected headphones. Tap Stop reading to stop.":
      "正在使用耳機朗讀。撳「停止朗讀」即可停止。",
    "Reading aloud through Bluetooth audio. Tap Stop reading to stop.":
      "正在使用藍牙音訊朗讀。撳「停止朗讀」即可停止。",
    "Reading aloud, but the phone’s volume is muted. Turn it up and try again.":
      "正在朗讀，但手機音量已調至最低。請調高音量再試。",
    "No sound: your iPhone media volume is at 0%. Press volume up and try again.":
      "沒有聲音：iPhone 媒體音量為 0%。請按音量增加鍵再試。",
    "Sound is going to Bluetooth audio.": "聲音會由藍牙裝置播放。",
    "Sound is going to headphones.": "聲音會由耳機播放。",
    "Sound is going to the iPhone speaker.": "聲音會由 iPhone 喇叭播放。",
    "The iPhone voice started.": "iPhone 語音已開始。",
    "iPhone media volume:": "iPhone 媒體音量：",
    "Media volume:": "媒體音量：",
    "Reading could not find an audio output. Check your volume and connected headphones.":
      "未能找到音訊輸出。請檢查音量及耳機連線。",
    "Getting ready to read.": "正在準備辨識。",
    "Reading the words.": "正在辨識文字。",
    "Finding the useful details.": "正在整理重要資料。",
    "Some words need a closer look.": "部分文字需要再核對。",
    "Here are the words we found.": "以下係辨識到嘅文字。",
    "No readable words were found. Try a closer photo in better light.":
      "未能辨識文字，請喺光線充足嘅地方影近啲。",
  };
  Object.assign(zh, {
    "Scans stay on this device. Clearing browser data can erase them.":
      "文件只存喺本機，清除瀏覽資料可能會刪走紀錄。",
    "Main navigation": "主選單",
    "Original document image": "文件原圖",
    "An installed English voice is available. Cantonese is available too. Test the voice below to check sound.":
      "已有離線英文同廣東話語音，可以試聽語音確認聲音。",
    "An installed English voice is available. Cantonese needs a separately installed voice. Test the voice below to check sound.":
      "已有離線英文語音，廣東話需要另外安裝語音，可以先試聽確認聲音。",

    "Enabling saving keeps this scan and future scans on this device. You can turn it off in preferences.":
      "啟用後會儲存呢份同之後掃描嘅文件，可以喺設定關閉。",
    "Opening saved scans…": "正在開啟掃描紀錄⋯",
    "Saved scan deleted.": "已刪除呢份儲存紀錄。",
    "All saved scans deleted.": "已刪除全部儲存紀錄。",
    "Scans that were never saved cannot be recovered automatically.":
      "從未儲存嘅文件，無法自動搵返。",
    "Original image is unavailable. Recognised text is available below.":
      "未能查看原圖，可以睇下面嘅辨識原文。",
    "Check your device": "檢查你的裝置",
    "A few checks for reading and listening on this device.":
      "檢查呢部裝置嘅閱讀同朗讀功能。",
    "Getting things ready": "正在準備",
    "Checking your reading files and available voices…":
      "正在檢查閱讀檔案同語音⋯",
    "Run device checks": "檢查裝置",
    "Ready to try reading": "可以開始辨識文件",
    "Some checks need attention": "部分項目需要處理",
    "Your reading files are available. Test the voice before listening.":
      "閱讀檔案已齊備，聆聽前可以先試聽語音。",
    "Reading is available. Install an offline English voice to listen.":
      "可以辨識文件，下載離線英文語音後可以聆聽。",
    "Review the steps below. Uploading a photo may still be available.":
      "請睇下面嘅處理方法，仍然可以嘗試上載相片。",
    "Available when you choose Scan": "揀掃描時可以使用",
    "Needs attention": "需要處理",
    "SightLine asks permission when you start scanning. No camera was opened by this check.":
      "開始掃描時先會要求相機權限，今次檢查冇開相機。",
    "Camera access is unavailable here. Upload a photo instead.":
      "未能使用相機，可以改為上載相片。",
    "Allow camera access in your device settings, or upload a photo.":
      "請喺裝置設定容許相機權限，或改為上載相片。",
    "Reading software": "辨識功能",
    Supported: "支援",
    "The browser supports on-device reading. Use a real scan to check recognition.":
      "瀏覽器支援本機辨識，請用實際文件檢查辨識效果。",
    "Offline reading": "離線閱讀",
    "Reading files saved": "已儲存閱讀檔案",
    "Not ready yet": "未準備好",
    "The required files are available on this device.":
      "所需檔案已存喺呢部裝置。",
    "Open SightLine online and wait for Ready offline. Then run this check again.":
      "連線開啟 SightLine，等顯示可離線使用後再檢查。",
    "Recognition files": "辨識檔案",
    Available: "可以使用",
    "Files missing": "缺少檔案",
    "The reading models and their runtime files are available. This is not an accuracy test.":
      "辨識模型及執行檔案齊備，呢項唔係準確度測試。",
    "Return to SightLine online to finish saving the reading files.":
      "請連線返回 SightLine 完成儲存閱讀檔案。",
    "Offline English voice available": "已有離線英文語音",
    "Try the voice": "試聽語音",
    "Test voice": "試聽語音",
    "Stop voice test": "停止試聽",
    "Turn up your volume. Check connected headphones too.":
      "請調高音量，亦可以檢查耳機連線。",
    "This check has not played any sound yet.": "呢項檢查未有播放語音。",
    "The voice is reading. Can you hear it?": "正在試聽，你聽到嗎？",
    "The voice test finished. Confirm below whether you heard it.":
      "試聽完成，請確認係咪聽到語音。",
    "Voice test stopped. You can try again.": "試聽已停止，可以再試。",
    "Did you hear the voice?": "你聽到語音嗎？",
    "I heard it": "我聽到",
    "I couldn’t hear it": "我聽唔到",
    "You confirmed that you heard the voice.": "你已確認聽到語音。",
    "Check the volume, silent mode and connected headphones. Then try Test voice again.":
      "請檢查音量、靜音模式同耳機連線，再試聽語音。",
    "These checks cannot confirm photo quality or speaker volume.":
      "呢啲檢查唔能夠確認相片質素或實際喇叭音量。",
    "Detailed reading checks": "詳細軟件檢查",
  });
  Object.assign(zh, {
    "Cantonese uses fixed summaries for supported, verified details. It does not translate arbitrary text. Medicine names stay as recognised. Chinese document extraction has not been validated. Check all medical information with a pharmacist.":
      "廣東話會用固定句式讀出支援同已核對嘅摘要，唔會翻譯任意文字，藥名保留原文。中文文件資料抽取未經驗證。醫療資料請向藥劑師核對。",
    "Take a photo. See the words more clearly, or hear them read aloud.":
      "影張相，可以放大睇清楚，或者聽朗讀。",
    "Your photos stay on your device.": "相片只留喺你嘅裝置。",
    "Keep a flat medicine label within the camera frame":
      "將平放嘅藥物標籤放喺鏡框內",
    "A clear photo makes a difference.": "相片清楚有幫助",
    "Keep the label flat, in good light, with all the words in view.":
      "標籤放平，光線要充足，全部文字都要入鏡。",
    Camera: "相機",
    "Bring the label into view.": "將標籤放入鏡框。",
    "Keep your hands steady and use good light.": "拎穩手機，並確保光線充足。",
    "Live camera preview": "即時相機畫面",
    "Your document photo": "你的文件相片",
    "Opening your camera…": "正在開啟相機⋯",
    "Retake photo": "重影相片",
    "Your photo stays private, right here on this device.":
      "相片只會喺呢部裝置處理，唔會上載。",
    "Document reading progress": "文件辨識進度",
    "Cancel reading": "取消辨識",
    "Directions couldn’t be matched to individual medicines. Check the original text and confirm them with a pharmacist.":
      "未能將用法配對至個別藥物，請核對原文，並同藥劑師確認。",
    "We haven’t filled in details we can’t read reliably. You can check the text below or try a clearer photo.":
      "未能清楚辨識嘅資料唔會填入，請核對下面嘅原文，或再影一張。",
    "We couldn’t organise this document into individual details. You can read the text below and compare it with the original.":
      "未能將文件整理成獨立資料，請睇下面嘅辨識文字，並核對原文。",
    "The reader can make mistakes. It cannot give medical advice. Confirm medicine names and directions with a pharmacist.":
      "辨識可能有錯誤，亦唔能夠提供醫療建議。藥名同用法請向藥劑師確認。",
    "Lay the label flat. Use bright, even light and keep all the words in view. Move closer if the text looks small.":
      "將標籤放平，用充足均勻嘅光線，全部文字都要入鏡。文字太細可以影近啲。",
    "Read-aloud isn’t available in this browser. You can still read the text here.":
      "呢個瀏覽器未能朗讀，你仍然可以睇辨識原文。",
    "No offline voice is available for this language. Download a voice in your device speech settings, then try again.":
      "未有呢種語言嘅離線語音，請喺裝置語音設定下載後再試。",
    "Tap Read Aloud again to allow speech to start.":
      "請再撳「朗讀」以啟動語音。",
    "History storage is unavailable in this browser.":
      "呢個瀏覽器未能儲存紀錄。",
    "History storage is blocked. Check browser privacy settings, or try the installed app.":
      "紀錄儲存被阻擋，請檢查瀏覽器私隱設定，或使用已安裝嘅程式。",
    "Storage is full. Delete older saved scans, or save without an image.":
      "儲存空間已滿，請刪除舊紀錄，或選擇唔儲存原圖。",
    "This saved scan is no longer available.": "呢份已儲存文件已不存在。",
    "History could not be saved. Your current scan is still available.":
      "未能儲存紀錄，目前嘅辨識結果仍然可以查看。",
  });
  const reverseLookup = new Map(Object.entries(zh).map(([en, cn]) => [cn, en]));
  function t(s) {
    const text = String(s ?? "");
    if (lang !== "zh-HK") return text;
    return zh[text] || text;
  }
  function apply(root = document.body) {
    if (!root) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (
        !node.parentElement ||
        node.parentElement.closest(
          "script,style,.technical-checks,.val,.original-value,#ocr-text,#summary-text,#cantonese,#result-meta,#source-image,.history-title",
        )
      )
        continue;
      const current = node.nodeValue,
        prior = sources.get(node);
      if (prior && current === prior.last && prior.language === lang) continue;
      const reverse = reverseLookup.get(current.trim());
      const source =
        prior && current === prior.last
          ? prior.source
          : reverse
            ? current.replace(current.trim(), reverse)
            : current;
      const key = source.trim().replace(/\s+/g, " ");
      const output =
        lang === "zh-HK" && zh[key]
          ? source.replace(source.trim(), zh[key])
          : source;
      sources.set(node, { source, last: output, language: lang });
      if (current !== output) node.nodeValue = output;
    }
    root.querySelectorAll("[alt]").forEach((el) => {
      const src = el.dataset.altSource || el.getAttribute("alt");
      el.dataset.altSource = src;
      if (el.getAttribute("alt") !== t(src)) el.setAttribute("alt", t(src));
    });
    root.querySelectorAll("[aria-label]").forEach((el) => {
      const src = el.dataset.labelSource || el.getAttribute("aria-label");
      el.dataset.labelSource = src;
      if (el.getAttribute("aria-label") !== t(src))
        el.setAttribute("aria-label", t(src));
    });
  }
  function set(value) {
    lang = value === "zh-HK" ? "zh-HK" : "en-GB";
    document.documentElement.lang = lang;
    document
      .querySelectorAll("[data-language]")
      .forEach((el) => (el.value = lang));
    apply();
  }
  function observe() {
    let queued = false;
    new MutationObserver(() => {
      if (queued) return;
      queued = true;
      queueMicrotask(() => {
        queued = false;
        apply();
      });
    }).observe(document.body, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }
  return {
    t,
    set,
    apply,
    observe,
    get language() {
      return lang;
    },
  };
})();
if (typeof window !== "undefined") window.SightLineI18n = SightLineI18n;
