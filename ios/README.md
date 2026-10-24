# SightLine for Xcode

Open `SightLine.xcodeproj`, select the **SightLine** scheme and an iPhone simulator,
then Run. For a physical iPhone, select your own development team in Signing &
Capabilities. No team, provisioning profile or credentials are stored here.

This is an iOS 17+ WKWebView shell around the same `../app` source. Models and
runtime are bundled; a read-only loopback server supplies the HTTP origin needed
by module workers. It does not listen on Wi-Fi. No hosted site is loaded.
The native bundle can load without an internet connection; browser service-worker
cache persistence is a separate capability that still needs device verification.

Regenerate after changing the project configuration: `xcodegen generate` from
this directory. App source changes are picked up by the next Xcode build.

An unsigned simulator build does not verify camera capture, VoiceOver, speech,
worker inference, memory limits or offline recovery on a physical iPhone.
