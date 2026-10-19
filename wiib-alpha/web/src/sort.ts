import { useMemo, useState } from 'react';

export type SortDir = 'asc' | 'desc';

export interface SortState<K extends string> {
  key: K;
  dir: SortDir;
}

type Value = number | string | null | undefined;

/** 点表头排序；空值不管升降都排最后。values 传模块级常量，别每次渲染新建 */
export function useSort<T, K extends string>(rows: T[], values: Record<K, (row: T) => Value>, initial: SortState<NoInfer<K>>) {
  const [sort, setSort] = useState<SortState<K>>(initial);

  const sorted = useMemo(() => {
    const get = values[sort.key];
    const sign = sort.dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const x = get(a);
      const y = get(b);
      if (x == null) return y == null ? 0 : 1;
      if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * sign;
    });
  }, [rows, values, sort]);

  // 换列时文字列先升序、数字列先降序
  const toggle = (key: K) => setSort(s => s.key === key
    ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' }
    : { key, dir: rows.length && typeof values[key](rows[0]) === 'string' ? 'asc' : 'desc' });

  return { sorted, sort, toggle };
}
