#!/bin/bash
set -eu
build_dir=$(mktemp -d /tmp/sightline-history-native.XXXXXX)
app_dir="$build_dir/SightLineHistoryAudit.app"
mkdir -p "$app_dir/Contents/MacOS"
cat > "$app_dir/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict><key>CFBundleIdentifier</key><string>com.sightline.reader.history-audit</string><key>CFBundleExecutable</key><string>SightLineHistoryAudit</string><key>CFBundlePackageType</key><string>APPL</string></dict></plist>
PLIST
swiftc ios/Sources/OfflineAssets.swift ios/Sources/WebAssetServer.swift app/tools/native_history/main.swift -o "$app_dir/Contents/MacOS/SightLineHistoryAudit"
"$app_dir/Contents/MacOS/SightLineHistoryAudit" "$PWD/app" write "$PWD/artifacts/bilingual-history/native-write.json" "$PWD/artifacts/aria-merge/sroie-X00016469612.jpg"
"$app_dir/Contents/MacOS/SightLineHistoryAudit" "$PWD/app" read "$PWD/artifacts/bilingual-history/native-read.json" "$PWD/artifacts/aria-merge/sroie-X00016469612.jpg"
