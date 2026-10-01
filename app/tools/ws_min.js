/**
 * Minimal RFC 6455 WebSocket client — just enough to speak CDP to a local
 * Chromium, with no npm dependency. Text frames only, which is all CDP uses.
 */
"use strict";

const net = require("net");
const crypto = require("crypto");
const { EventEmitter } = require("events");

class WsClient extends EventEmitter {
  constructor(sock) {
    super();
    this.sock = sock;
    this.buf = Buffer.alloc(0);
    this.closed = false;
    sock.on("data", (d) => this._onData(d));
    sock.on("close", () => { this.closed = true; });
    sock.on("error", (e) => this.emit("error", e));
  }

  _onData(chunk) {
    this.buf = Buffer.concat([this.buf, chunk]);
    for (;;) {
      if (this.buf.length < 2) return;
      const b0 = this.buf[0], b1 = this.buf[1];
      const opcode = b0 & 0x0f;
      const masked = (b1 & 0x80) === 0x80;
      let len = b1 & 0x7f;
      let off = 2;
      if (len === 126) {
        if (this.buf.length < 4) return;
        len = this.buf.readUInt16BE(2);
        off = 4;
      } else if (len === 127) {
        if (this.buf.length < 10) return;
        len = Number(this.buf.readBigUInt64BE(2));
        off = 10;
      }
      let mask = null;
      if (masked) {
        if (this.buf.length < off + 4) return;
        mask = this.buf.subarray(off, off + 4);
        off += 4;
      }
      if (this.buf.length < off + len) return;
      const payload = Buffer.from(this.buf.subarray(off, off + len));
      if (mask) for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
      this.buf = this.buf.subarray(off + len);

      if (opcode === 0x1) this.emit("message", payload.toString("utf8"));
      else if (opcode === 0x8) { this.closed = true; try { this.sock.end(); } catch (e) { /* closing */ } return; }
      else if (opcode === 0x9) { this._frame(0xA, payload); }  // pong
      else if (opcode === 0xA) { /* ping reply ignored */ }
    }
  }

  _frame(opcode, payload) {
    const data = Buffer.isBuffer(payload) ? payload : Buffer.from(String(payload), "utf8");
    const len = data.length;
    let header;
    if (len < 126) {
      header = Buffer.alloc(6);
      header[1] = 0x80 | len;
    } else if (len < 65536) {
      header = Buffer.alloc(8);
      header[1] = 0x80 | 126;
      header.writeUInt16BE(len, 2);
    } else {
      header = Buffer.alloc(14);
      header[1] = 0x80 | 127;
      header.writeBigUInt64BE(BigInt(len), 2);
    }
    header[0] = 0x80 | opcode;
    const mask = crypto.randomBytes(4);
    mask.copy(header, header.length - 4);
    const masked = Buffer.from(data);
    for (let i = 0; i < masked.length; i++) masked[i] ^= mask[i & 3];
    this.sock.write(Buffer.concat([header, masked]));
  }

  send(str) { if (!this.closed) this._frame(0x1, str); }
  close() { try { this.sock.end(); } catch (e) { /* already gone */ } }
}

WsClient.connect = function connect(url) {
  const u = new URL(url);
  return new Promise((resolve, reject) => {
    const key = crypto.randomBytes(16).toString("base64");
    const sock = net.connect({ host: u.hostname, port: Number(u.port || 80) }, () => {
      sock.write(
        "GET " + u.pathname + u.search + " HTTP/1.1\r\n"
        + "Host: " + u.host + "\r\n"
        + "Upgrade: websocket\r\n"
        + "Connection: Upgrade\r\n"
        + "Sec-WebSocket-Key: " + key + "\r\n"
        + "Sec-WebSocket-Version: 13\r\n\r\n"
      );
    });
    sock.on("error", reject);
    let handshook = false;
    let acc = Buffer.alloc(0);
    const onData = (d) => {
      acc = Buffer.concat([acc, d]);
      const i = acc.indexOf("\r\n\r\n");
      if (i === -1) return;
      const head = acc.subarray(0, i).toString();
      if (!/101/.test(head.split("\r\n")[0])) {
        sock.destroy();
        return reject(new Error("websocket upgrade failed: " + head.split("\r\n")[0]));
      }
      handshook = true;
      const rest = acc.subarray(i + 4);
      sock.removeListener("data", onData);
      const c = new WsClient(sock);
      if (rest.length) c._onData(rest);
      resolve(c);
    };
    sock.on("data", onData);
    setTimeout(() => { if (!handshook) { sock.destroy(); reject(new Error("websocket connect timeout")); } }, 15000);
  });
};

module.exports = WsClient;
module.exports.connect = WsClient.connect;
module.exports.default = WsClient;
