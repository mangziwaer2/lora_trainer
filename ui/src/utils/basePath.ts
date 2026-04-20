const rawBasePath = process.env.NEXT_PUBLIC_UI_BASE_PATH || '';

export const uiBasePath =
  rawBasePath && rawBasePath !== '/' ? rawBasePath.replace(/\/+$/, '') : '';

export function withBasePath(path: string): string {
  if (!path) {
    return uiBasePath || '/';
  }

  if (/^(?:[a-z]+:)?\/\//i.test(path)) {
    return path;
  }

  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${uiBasePath}${normalizedPath}`;
}
