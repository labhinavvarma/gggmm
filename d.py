/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable standalone output for Docker
  output: 'standalone',
  
  // Optional: Reduce image size by disabling source maps in production
  productionBrowserSourceMaps: false,
  
  // Optional: Optimize images
  images: {
    unoptimized: false,
  },
  
  // Handle React 19 experimental features if needed
  experimental: {
    // Add any experimental features here
  }
}

module.exports = nextConfig
