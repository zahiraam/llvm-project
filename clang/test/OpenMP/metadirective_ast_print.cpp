// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 \
// RUN: -triple x86_64-unknown-linux-gnu -ast-print %s -o - | FileCheck %s 

// expected-no-diagnostics

void bar();
void test_nonconstant_condition(bool use_gpu) {
  #pragma omp metadirective \
      when(user={condition(use_gpu)}: parallel) \
      otherwise(single)
  {
       bar();
  }
}

// LABEL: void test_nonconstant_condition(bool use_gpu)
// CHECK: #pragma omp metadirective when(use_gpu: #pragma omp parallel) otherwise( #pragma omp single)
// CHECK: bar();


