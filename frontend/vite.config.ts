import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发服务器把 /v1 代理到的后端地址；后端起在别的端口时用环境变量覆盖，
// 例如 FINHARNESS_API_ORIGIN=http://127.0.0.1:8000 npm run dev。
const apiOrigin = process.env.FINHARNESS_API_ORIGIN ?? "http://127.0.0.1:8001";

export default defineConfig({
  plugins: [react()],
  build: {
    cssMinify: "esbuild",
    rolldownOptions: {
      output: {
        // 默认全部打进单个入口块（约 960 kB）：antd、markdown 渲染与 React
        // 运行时和应用代码捆在一起，任何一处改动都会让整包缓存失效。
        // 这里按依赖族拆成稳定 vendor 块，应用代码单独走入口块。
        // rolldown 分组会递归捕获依赖，因此 react 运行时随最先用到它的
        // vendor 组（cssinjs/antd）走、不单独成块；末位的 vendor 组兜底
        // 接住其余 node_modules 依赖，入口块只保留应用代码。
        // 注意：不能设 includeDependenciesRecursively: false——跨块循环依赖
        // 会因执行顺序错乱而在运行时报 "d is not a function"，应用白屏。
        codeSplitting: {
          groups: [
            {
              name: "vendor-icons",
              test: /[\\/]node_modules[\\/]@ant-design[\\/](icons|icons-svg)/,
            },
            {
              name: "vendor-cssinjs",
              test: /[\\/]node_modules[\\/]@ant-design[\\/](cssinjs|cssinjs-utils|fast-color|colors)/,
            },
            {
              name: "vendor-antd",
              test: /[\\/]node_modules[\\/](antd|@rc-component|rc-[a-z0-9-]+)/,
            },
            {
              name: "vendor-markdown",
              test: /[\\/]node_modules[\\/](react-markdown|remark|unified|unist|mdast|micromark|hast|vfile|parse5|property-information|space-separated-tokens|comma-separated-tokens|character-entities|decode-named-character-reference|html-url-attributes|inline-style-parser|style-to-object|zwitch|bail|trough|devlop|is-plain-obj|longest-streak)/,
            },
            { name: "vendor", test: /[\\/]node_modules[\\/]/ },
          ],
        },
      },
    },
  },
  server: {
    port: 5174,
    proxy: {
      "/v1": apiOrigin,
    },
  },
});
