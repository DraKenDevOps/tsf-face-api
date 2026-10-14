import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';
import typescript from '@rollup/plugin-typescript';

const { minify } = process.env;

export default {
  input: 'src/index.ts',
  plugins: [
    typescript({
      tsconfig: './tsconfig.json',
      compilerOptions: {
        module: 'ESNext',
        target: 'ES2020',
        sourceMap: !minify,
        declaration: false,
        declarationMap: false,
        outDir: 'dist'
      }
    }),
    resolve({
      browser: true,
      preferBuiltins: false
    }),
    commonjs({
      include: 'node_modules/**'
    }),
    ...(minify ? [terser()] : [])
  ],
  output: {
    extend: true,
    file: `dist/ts-face-api${minify ? '.min' : ''}.js`,
    format: 'umd',
    name: 'faceapi',
    globals: {
      crypto: 'crypto',
      fs: 'fs',
      path: 'path'
    },
    sourcemap: !minify
  },
  external: ['crypto', 'fs', 'path'],
  onwarn: (warning) => {
    const ignoreWarnings = [
      'CIRCULAR_DEPENDENCY',
      'CIRCULAR',
      'THIS_IS_UNDEFINED'
    ];
    if (ignoreWarnings.includes(warning.code)) {
      return;
    }
    if (warning.missing === 'alea') {
      return;
    }
    console.warn(warning.message);
  }
};
