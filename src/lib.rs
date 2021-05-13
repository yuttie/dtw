use std::vec::Vec;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use ndarray::Ix2;
use numpy::{PyArray, ToPyArray};

#[pymodule]
fn dtw(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dp, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_path, m)?)?;
    m.add_function(wrap_pyfunction!(dist, m)?)?;
    m.add_function(wrap_pyfunction!(pdist, m)?)?;

    Ok(())
}

#[pyfunction]
fn dp<'py>(py: Python<'py>, s: Vec<usize>, t: Vec<usize>, d: &'py PyArray<u32, Ix2>) -> &'py PyArray<u32, Ix2> {
    let d = d.readonly();
    let d = d.as_array();
    py.allow_threads(|| rs::dp(&s, &t, &d)).to_pyarray(py)
}

#[pyfunction]
fn optimal_path<'py>(py: Python<'py>, table: &'py PyArray<u32, Ix2>) -> Vec<(usize, usize)> {
    let table = table.readonly();
    let table = table.as_array();
    py.allow_threads(|| rs::optimal_path(&table))
}

#[pyfunction]
fn dist<'py>(py: Python<'py>, s: Vec<usize>, t: Vec<usize>, d: &'py PyArray<u32, Ix2>) -> u32 {
    let d = d.readonly();
    let d = d.as_array();
    py.allow_threads(|| rs::dist(&s, &t, &d))
}

#[pyfunction]
fn pdist<'py>(py: Python<'py>, xs: Vec<Vec<usize>>, d: &'py PyArray<u32, Ix2>) -> &'py PyArray<u32, Ix2> {
    let d = d.readonly();
    let d = d.as_array();
    py.allow_threads(|| rs::pdist(&xs, &d)).to_pyarray(py)
}

mod rs {
    use std::cmp::min;
    use std::vec::Vec;
    use ndarray::{ArrayBase, Array2, Ix2, Data, RawData};
    use ndarray::parallel::prelude::par_azip;

    /// Compute a DP table.
    pub fn dp<'a, S: Sync + Data + RawData<Elem = u32>>(s: &[usize], t: &[usize], d: &ArrayBase<S, Ix2>) -> Array2<u32> {
        let m = s.len();
        let n = t.len();
        let mut table = Array2::<u32>::zeros((m + 1, n + 1));

        table[[0, 0]] = 0;

        {
            let i = 1;
            let j = 1;
            let cost = d[[s[i - 1], t[j - 1]]];
            table[[i, j]] = cost + table[[i - 1, j - 1]];
        }

        for i in 2..=m {
            let j = 1;
            let cost = d[[s[i - 1], t[j - 1]]];
            table[[i, j]] = cost + table[[i - 1, j]];
        }

        for j in 2..=n {
            let i = 1;
            let cost = d[[s[i - 1], t[j - 1]]];
            table[[i, j]] = cost + table[[i, j - 1]];
        }

        for i in 2..=m {
            for j in 2..=n {
                let cost = d[[s[i - 1], t[j - 1]]];
                table[[i, j]] = cost + min(min(table[[i - 1, j]], table[[i, j - 1]]), table[[i - 1, j - 1]]);
            }
        }

        table
    }

    /// Compute the optimal path from a given DP table.
    pub fn optimal_path<'a, S: Sync + Data + RawData<Elem = u32>>(table: &ArrayBase<S, Ix2>) -> Vec<(usize, usize)> {
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut i = table.shape()[0] - 1;
        let mut j = table.shape()[1] - 1;
        while i >= 2 && j >= 2 {
            path.push((i - 1, j - 1));
            if table[[i - 1, j - 1]] <= table[[i, j - 1]] {
                if table[[i - 1, j - 1]] <= table[[i - 1, j]] {
                    i -= 1;
                    j -= 1;
                }
                else {
                    i -= 1;
                }
            }
            else {
                if table[[i, j - 1]] <= table[[i - 1, j]] {
                    j -= 1;
                }
                else {
                    i -= 1;
                }
            }
        }
        while i >= 2 {
            path.push((i - 1, j - 1));
            i -= 1;
        }
        while j >= 2 {
            path.push((i - 1, j - 1));
            j -= 1;
        }
        path.push((i - 1, j - 1));
        path.reverse();
        path
    }

    /// Compute the distance between two given sequences.
    pub fn dist<'a, S: Sync + Data + RawData<Elem = u32>>(s: &[usize], t: &[usize], d: &ArrayBase<S, Ix2>) -> u32 {
        let m = s.len();
        let n = t.len();
        let table = dp(s, t, &d);
        table[[m, n]]
    }

    /// Compute pairwise distances between sequences.
    pub fn pdist<'a, T: AsRef<[usize]> + Sync, S: Sync + Send + Data + RawData<Elem = u32>>(xs: &[T], d: &ArrayBase<S, Ix2>) -> Array2<u32> {
        let n = xs.len();
        let mut dmat = Array2::<u32>::zeros((n, n));
        let indices = ndarray::indices_of(&dmat);
        par_azip!((dmat_ij in &mut dmat, (i, j) in indices) {
            *dmat_ij = dist(xs[i].as_ref(), xs[j].as_ref(), &d);
        });
        dmat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_dp() {
        let d: Array2<u32> = array![
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0],
        ];

        let s = vec![0, 1, 2, 3, 4];
        let t = vec![0, 1, 2, 3, 4];
        assert_eq!(rs::dp(&s, &t, &d), array![
            [0,  0, 0, 0, 0,  0],
            [0,  0, 1, 3, 6, 10],
            [0,  1, 0, 1, 3,  6],
            [0,  3, 1, 0, 1,  3],
            [0,  6, 3, 1, 0,  1],
            [0, 10, 6, 3, 1,  0],
        ]);
    }

    #[test]
    fn test_optimal_path() {
        let d = array![
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0],
        ];

        let s = vec![0, 1, 2, 3, 4];
        let t = vec![0, 1, 2, 3, 4];
        let table = rs::dp(&s, &t, &d);
        assert_eq!(rs::optimal_path(&table), vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
        ]);
    }

    #[test]
    fn test_dist() {
        let d = array![
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0],
        ];

        let s = vec![0, 1, 2, 3, 4];
        let t = vec![0, 1, 2, 3, 4];
        assert_eq!(rs::dist(&s, &t, &d), 0);
    }
}
