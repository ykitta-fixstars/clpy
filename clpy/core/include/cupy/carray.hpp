#pragma once

// CArray
template <typename T, int _ndim>
class CArray {
public:
  static const int ndim = _ndim;
  int size() const;

  const ptrdiff_t* shape() const;

  const ptrdiff_t* strides() const;

  template <typename Int>
  T& operator[](const Int (&idx)[ndim]);

  template <typename Int>
  const T& operator[](const Int (&idx)[ndim]) const;

  T& operator[](ptrdiff_t i);

  const T& operator[](ptrdiff_t i) const;
};

template <int _ndim>
class CIndexer {
public:
  static const int ndim = _ndim;
  ptrdiff_t size() const;

  void set(ptrdiff_t i);

  const ptrdiff_t* get() const;
};

