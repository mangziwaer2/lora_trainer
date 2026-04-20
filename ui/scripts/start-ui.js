const http = require('http');
const { spawn } = require('child_process');
const path = require('path');

const PORT = Number(process.env.PORT || 8675);

function requestRoot(port) {
  return new Promise(resolve => {
    const req = http.get(
      {
        host: '127.0.0.1',
        port,
        path: '/',
        timeout: 1500,
      },
      res => {
        res.resume();
        resolve({ ok: true, statusCode: res.statusCode });
      },
    );

    req.on('timeout', () => {
      req.destroy();
      resolve({ ok: false });
    });

    req.on('error', () => resolve({ ok: false }));
  });
}

async function main() {
  const probe = await requestRoot(PORT);
  if (probe.ok) {
    console.log(`[UI] Port ${PORT} is already serving HTTP (status ${probe.statusCode}). Reusing existing UI instance.`);
    return;
  }

  const nextBin = path.join('node_modules', 'next', 'dist', 'bin', 'next');
  const child = spawn(process.execPath, [nextBin, 'start', '--port', String(PORT)], {
    stdio: 'inherit',
    shell: false,
  });

  child.on('exit', code => {
    process.exit(code ?? 0);
  });
}

main().catch(error => {
  console.error('[UI] Failed to start UI:', error);
  process.exit(1);
});
