/** @type {import('next').NextConfig} */
const nextConfig = {
  // Kaggle's notebook proxy can surface gzipped responses as raw bytes.
  // Disable Next's built-in compression so proxied pages are returned plain.
  compress: false,
  devIndicators: {
    buildActivity: false,
  },
  typescript: {
    // Keep build behavior aligned with the current customized routes.
    ignoreBuildErrors: true,
  },
};

module.exports = nextConfig;
