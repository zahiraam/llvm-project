// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu \
// RUN: -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// RUN %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu \
// RUN -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

struct foo {
  foo(int j) : i(j) {};
  int i;
};

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 545]
// CHECK: @.offload_maptypes.2 = private unnamed_addr constant [1 x i64] [i64 547]
// CHECK: @.offload_maptypes.4 = private unnamed_addr constant [1 x i64] [i64 547]
// CHECK: @.offload_maptypes.6 = private unnamed_addr constant [1 x i64] [i64 545]
// CHECK: @.offload_maptypes.8 = private unnamed_addr constant [1 x i64] [i64 547]
// CHECK: @.offload_maptypes.10 = private unnamed_addr constant [1 x i64] [i64 545]
// CHECK: @.offload_maptypes.12 = private unnamed_addr constant [1 x i64] [i64 35]
// CHECK: @.offload_maptypes.14 = private unnamed_addr constant [2 x i64] [i64 545, i64 547]

// Const struct, no mutable members -> mapped 'to'

// LABEL: test_const_no_mutable
// CHECK: store ptr @.offload_maptypes, ptr {{.*}}, align 8
void test_const_no_mutable() {
  const foo a(2);
#pragma omp target
  {
    int x = a.i;
  }
}

// Non-const -> mapped 'tofrom'

// LABEL: define dso_local void @_Z13test_nonconstv
// CHECK: store ptr @.offload_maptypes.2, ptr {{.*}}, align 8
void test_nonconst() {
  foo a(2);
#pragma omp target
  {
    int x = a.i;
  }
}

struct foo_mutable {
  foo_mutable(int j) : i(j), m(0) {};
  int i;
  mutable int m;
};

// Const struct with a mutable member -> mapped 'tofrom'

// LABEL: define dso_local void @_Z23test_const_with_mutablev
// CHECK: store ptr @.offload_maptypes.4, ptr {{.*}}, align 8
void test_const_with_mutable() {
  const foo_mutable a(2);
#pragma omp target
  {
    a.m = 1;
  }
}

struct foo_nested {
  foo_nested(int j) : inner(j), z(j) {};
  foo inner;
  const int z;
};

// Const struct nested inside another const struct -> mapped 'to'

// LABEL: define dso_local void @_Z17test_const_nestedv() #0 {
// CHECK: store ptr @.offload_maptypes.6, ptr {{.*}}, align 8
void test_const_nested() {
  const foo_nested a(2);
#pragma omp target
  {
    int x = a.inner.i;
  }
}

struct foo_nested_mutable {
  foo_nested_mutable(int j) : inner(j), z(j) {};
  foo_mutable inner; // has mutable member buried inside
  const int z;
};

// Const struct nested inside another const struct, where the nested
// struct has a mutable member -> mapped 'tofrom'

// LABEL: define dso_local void @_Z30test_const_nested_with_mutablev
// CHECK: store ptr @.offload_maptypes.8, ptr {{.*}}, align 8
void test_const_nested_with_mutable() {
  const foo_nested_mutable a(2);
#pragma omp target
  {
    a.inner.m = 1;
  }
}

// Const array of foo -> mapped 'to'

// LABEL: define dso_local void @_Z16test_const_arrayv
// CHECK: store ptr @.offload_maptypes.10, ptr {{.*}}, align 8
void test_const_array() {
  const foo arr[4] = {1, 2, 3, 4};
#pragma omp target
  {
    int x = arr[0].i;
  }
}

// Explicit map(tofrom:) on a const struct -> mapped 'tofrom'

// LABEL: define dso_local void @_Z27test_explicit_map_overridesv
// CHECK: store ptr @.offload_maptypes.12, ptr {{.*}}, align 8
void test_explicit_map_overrides() {
  const foo a(2);
#pragma omp target map(tofrom:a)
  {
    int x = a.i;
  }
}

// Mixed: const foo (to) and non-const foo (tofrom) in the same region.

// LABEL: define dso_local void @_Z10test_mixedv
// CHECK: store ptr @.offload_maptypes.14, ptr {{.*}}, align 8
void test_mixed() {
  const foo ca(2);
  foo ma(3);
#pragma omp target
  {
    int x = ca.i;
    ma.i = 99;
  }
}

// Defaultmap(tofrom:aggregate) explicit -> mapped 'to'.

// LABEL: define dso_local void @_Z31test_defaultmap_tofrom_explicitv
// CHECK: store ptr @.offload_maptypes.16, ptr {{.*}}, align 8
void test_defaultmap_tofrom_explicit() {
  const foo a(2);
#pragma omp target defaultmap(tofrom:aggregate)
  {
    int x = a.i;
  }
}
