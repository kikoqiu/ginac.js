import typescript from '@rollup/plugin-typescript';
import commonjs from '@rollup/plugin-commonjs';
import copy from 'rollup-plugin-copy';
import resolve from '@rollup/plugin-node-resolve';



export default {
  input: 'src/index.ts',
  output: [
    {
      file: 'dist/index.umd.js',
      name: 'ginac',
      format: 'umd',
      sourcemap: 'inline',
    },
    {
      file: 'dist/index.esm.js',
      format: 'es',
      sourcemap: 'inline',
    },
  ],
  external: ['./src/ginac.js'],  
  plugins: [
    resolve(),
    copy({
      targets: [
        { src: './ginac/build/bind/ginac.wasm', dest: './dist' },
        { src: './ginac/build/bind/ginac.js', dest: './dist' }
      ],
    }),
    commonjs(),
    typescript(),
  ],
};

