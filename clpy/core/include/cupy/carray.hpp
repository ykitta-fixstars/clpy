#pragma once

#ifdef __ULTIMA
#include<clpy/carray.clh>
#endif

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

#ifdef __ULTIMA
namespace __ultima_detail{
template<int>struct cindexer_;
template<>struct cindexer_<0>{typedef CIndexer_0 type;};
template<>struct cindexer_<1>{typedef CIndexer_1 type;};
template<>struct cindexer_<2>{typedef CIndexer_2 type;};
template<>struct cindexer_<3>{typedef CIndexer_3 type;};
template<>struct cindexer_<4>{typedef CIndexer_4 type;};
template<>struct cindexer_<5>{typedef CIndexer_5 type;};
template<>struct cindexer_<6>{typedef CIndexer_6 type;};
template<>struct cindexer_<7>{typedef CIndexer_7 type;};
template<>struct cindexer_<8>{typedef CIndexer_8 type;};
template<>struct cindexer_<9>{typedef CIndexer_9 type;};
template<>struct cindexer_<10>{typedef CIndexer_10 type;};
template<>struct cindexer_<11>{typedef CIndexer_11 type;};
template<>struct cindexer_<12>{typedef CIndexer_12 type;};
template<>struct cindexer_<13>{typedef CIndexer_13 type;};
template<>struct cindexer_<14>{typedef CIndexer_14 type;};
template<>struct cindexer_<15>{typedef CIndexer_15 type;};
}
#endif

template <int _ndim>
class CIndexer {
public:
  static const int ndim = _ndim;
  ptrdiff_t size() const;

  void set(ptrdiff_t i);

  const ptrdiff_t* get() const;

#ifdef __ULTIMA
  const typename __ultima_detail::cindexer_<_ndim>::type* operator&()const;
#endif
};

