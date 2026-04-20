const rawBasePath = process.env.NEXT_PUBLIC_UI_BASE_PATH || '';
const normalizedBasePath =
  rawBasePath && rawBasePath !== '/' ? rawBasePath.replace(/\/+$/, '') : '';

if (normalizedBasePath && !normalizedBasePath.startsWith('/')) {
  throw new Error('NEXT_PUBLIC_UI_BASE_PATH must start with "/" when provided.');
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Kaggle's notebook proxy can surface gzipped responses as raw bytes.
  // Disable Next's built-in compression so proxied pages are returned plain.
  compress: false,
  ...(normalizedBasePath ? { basePath: normalizedBasePath } : {}),
  devIndicators: {
    buildActivity: false,
  },
  typescript: {
    // Keep build behavior aligned with the current customized routes.
    ignoreBuildErrors: true,
  },
};

module.exports = nextConfig;
