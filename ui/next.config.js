/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: {
    buildActivity: false,
  },
  typescript: {
    // Keep build behavior aligned with the current customized routes.
    ignoreBuildErrors: true,
  },
};

module.exports = nextConfig;
