/**
 * Minimal Chrome DevTools Protocol client (shared by the browser harnesses).
 *
 * No npm dependency: ws_min.js implements just enough RFC 6455 to talk to a
 * local Chromium, and this wraps it with request/response correlation.
 */
"use strict";

const http = require("http");
const WsClient = require("./ws_min.js");

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.events = [];
  }

  static async attach(port) {
    const list = await new Promise((res, rej) => {
      http.get({ host: "127.0.0.1", port, path: "/json/list" }, (r) => {
        let b = "";
        r.on("data", (c) => (b += c));
        r.on("end", () => res(JSON.parse(b)));
      }).on("error", rej);
    });
    const page = list.find((t) => t.type === "page");
    if (!page) throw new Error("no page target");
    const ws = await WsClient.connect(page.webSocketDebuggerUrl);
    const cdp = new CDP(ws);
    ws.on("message", (msg) => {
      let m;
      try { m = JSON.parse(msg); } catch (e) { return; }
      if (m.id && cdp.pending.has(m.id)) {
        const { resolve, reject } = cdp.pending.get(m.id);
        cdp.pending.delete(m.id);
        if (m.error) reject(new Error(JSON.stringify(m.error)));
        else resolve(m.result);
      } else if (m.method) {
        cdp.events.push(m);
      }
    });
    return cdp;
  }

  send(method, params, timeoutMs) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params: params || {} }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error("CDP timeout: " + method));
        }
      }, timeoutMs || 60000);
    });
  }

  /** Evaluate an expression in the page. Awaits promises. */
  async eval(expression, timeoutMs) {
    const r = await this.send("Runtime.evaluate", {
      expression, returnByValue: true, awaitPromise: true,
    }, timeoutMs);
    if (r.exceptionDetails) {
      const d = r.exceptionDetails;
      throw new Error("eval threw: " + ((d.exception && d.exception.description) || d.text));
    }
    return r.result.value;
  }

  close() { try { this.ws.close(); } catch (e) { /* already closed */ } }
}

module.exports = { CDP, attach: CDP.attach };
