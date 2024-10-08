; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -passes=instcombine -S %s | FileCheck %s

define i32 @reassoc_add_nuw(i32 %x) {
; CHECK-LABEL: @reassoc_add_nuw(
; CHECK-NEXT:    [[ADD1:%.*]] = add nuw i32 [[X:%.*]], 68
; CHECK-NEXT:    ret i32 [[ADD1]]
;
  %add0 = add nuw i32 %x, 4
  %add1 = add nuw i32 %add0, 64
  ret i32 %add1
}

; This does the wrong thing because the sub is turned into an add of a
; negative constant first which drops the nuw.
define i32 @reassoc_sub_nuw(i32 %x) {
; CHECK-LABEL: @reassoc_sub_nuw(
; CHECK-NEXT:    [[SUB1:%.*]] = add i32 [[X:%.*]], -68
; CHECK-NEXT:    ret i32 [[SUB1]]
;
  %sub0 = sub nuw i32 %x, 4
  %sub1 = sub nuw i32 %sub0, 64
  ret i32 %sub1
}

define i32 @reassoc_mul_nuw(i32 %x) {
; CHECK-LABEL: @reassoc_mul_nuw(
; CHECK-NEXT:    [[MUL1:%.*]] = mul nuw i32 [[X:%.*]], 260
; CHECK-NEXT:    ret i32 [[MUL1]]
;
  %mul0 = mul nuw i32 %x, 4
  %mul1 = mul nuw i32 %mul0, 65
  ret i32 %mul1
}

define i32 @no_reassoc_add_nuw_none(i32 %x) {
; CHECK-LABEL: @no_reassoc_add_nuw_none(
; CHECK-NEXT:    [[ADD1:%.*]] = add i32 [[X:%.*]], 68
; CHECK-NEXT:    ret i32 [[ADD1]]
;
  %add0 = add i32 %x, 4
  %add1 = add nuw i32 %add0, 64
  ret i32 %add1
}

define i32 @no_reassoc_add_none_nuw(i32 %x) {
; CHECK-LABEL: @no_reassoc_add_none_nuw(
; CHECK-NEXT:    [[ADD1:%.*]] = add i32 [[X:%.*]], 68
; CHECK-NEXT:    ret i32 [[ADD1]]
;
  %add0 = add nuw i32 %x, 4
  %add1 = add i32 %add0, 64
  ret i32 %add1
}

define i32 @reassoc_x2_add_nuw(i32 %x, i32 %y) {
; CHECK-LABEL: @reassoc_x2_add_nuw(
; CHECK-NEXT:    [[ADD1:%.*]] = add nuw i32 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[ADD2:%.*]] = add nuw i32 [[ADD1]], 12
; CHECK-NEXT:    ret i32 [[ADD2]]
;
  %add0 = add nuw i32 %x, 4
  %add1 = add nuw i32 %y, 8
  %add2 = add nuw i32 %add0, %add1
  ret i32 %add2
}

define i32 @reassoc_x2_mul_nuw(i32 %x, i32 %y) {
; CHECK-LABEL: @reassoc_x2_mul_nuw(
; CHECK-NEXT:    [[MUL1:%.*]] = mul i32 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[MUL2:%.*]] = mul nuw i32 [[MUL1]], 45
; CHECK-NEXT:    ret i32 [[MUL2]]
;
  %mul0 = mul nuw i32 %x, 5
  %mul1 = mul nuw i32 %y, 9
  %mul2 = mul nuw i32 %mul0, %mul1
  ret i32 %mul2
}

define i32 @reassoc_x2_sub_nuw(i32 %x, i32 %y) {
; CHECK-LABEL: @reassoc_x2_sub_nuw(
; CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[SUB2:%.*]] = add i32 [[TMP1]], 4
; CHECK-NEXT:    ret i32 [[SUB2]]
;
  %sub0 = sub nuw i32 %x, 4
  %sub1 = sub nuw i32 %y, 8
  %sub2 = sub nuw i32 %sub0, %sub1
  ret i32 %sub2
}

define i32 @tryFactorization_add_nuw_mul_nuw(i32 %x) {
; CHECK-LABEL: @tryFactorization_add_nuw_mul_nuw(
; CHECK-NEXT:    [[ADD2:%.*]] = shl nuw i32 [[X:%.*]], 2
; CHECK-NEXT:    ret i32 [[ADD2]]
;
  %mul1 = mul nuw i32 %x, 3
  %add2 = add nuw i32 %mul1, %x
  ret i32 %add2
}

define i32 @tryFactorization_add_nuw_mul_nuw_int_max(i32 %x) {
; CHECK-LABEL: @tryFactorization_add_nuw_mul_nuw_int_max(
; CHECK-NEXT:    [[ADD2:%.*]] = shl nuw i32 [[X:%.*]], 31
; CHECK-NEXT:    ret i32 [[ADD2]]
;
  %mul1 = mul nuw i32 %x, 2147483647
  %add2 = add nuw i32 %mul1, %x
  ret i32 %add2
}

define i32 @tryFactorization_add_mul_nuw(i32 %x) {
; CHECK-LABEL: @tryFactorization_add_mul_nuw(
; CHECK-NEXT:    [[ADD2:%.*]] = shl i32 [[X:%.*]], 2
; CHECK-NEXT:    ret i32 [[ADD2]]
;
  %mul1 = mul i32 %x, 3
  %add2 = add nuw i32 %mul1, %x
  ret i32 %add2
}

define i32 @tryFactorization_add_nuw_mul(i32 %x) {
; CHECK-LABEL: @tryFactorization_add_nuw_mul(
; CHECK-NEXT:    [[ADD2:%.*]] = shl i32 [[X:%.*]], 2
; CHECK-NEXT:    ret i32 [[ADD2]]
;
  %mul1 = mul nuw i32 %x, 3
  %add2 = add i32 %mul1, %x
  ret i32 %add2
}

define i32 @tryFactorization_add_nuw_mul_nuw_mul_nuw_var(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @tryFactorization_add_nuw_mul_nuw_mul_nuw_var(
; CHECK-NEXT:    [[MUL21:%.*]] = add i32 [[Y:%.*]], [[Z:%.*]]
; CHECK-NEXT:    [[ADD1:%.*]] = mul nuw i32 [[X:%.*]], [[MUL21]]
; CHECK-NEXT:    ret i32 [[ADD1]]
;
  %mul1 = mul nuw i32 %x, %y
  %mul2 = mul nuw i32 %x, %z
  %add1 = add nuw i32 %mul1, %mul2
  ret i32 %add1
}

define i32 @tryFactorization_add_nuw_mul_mul_nuw_var(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @tryFactorization_add_nuw_mul_mul_nuw_var(
; CHECK-NEXT:    [[MUL21:%.*]] = add i32 [[Y:%.*]], [[Z:%.*]]
; CHECK-NEXT:    [[ADD1:%.*]] = mul i32 [[X:%.*]], [[MUL21]]
; CHECK-NEXT:    ret i32 [[ADD1]]
;
  %mul1 = mul i32 %x, %y
  %mul2 = mul nuw i32 %x, %z
  %add1 = add nuw i32 %mul1, %mul2
  ret i32 %add1
}

define i32 @tryFactorization_add_nuw_mul_nuw_mul_var(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @tryFactorization_add_nuw_mul_nuw_mul_var(
; CHECK-NEXT:    [[MUL21:%.*]] = add i32 [[Y:%.*]], [[Z:%.*]]
; CHECK-NEXT:    [[ADD1:%.*]] = mul i32 [[X:%.*]], [[MUL21]]
; CHECK-NEXT:    ret i32 [[ADD1]]
;
  %mul1 = mul nuw i32 %x, %y
  %mul2 = mul i32 %x, %z
  %add1 = add nuw i32 %mul1, %mul2
  ret i32 %add1
}

define i32 @tryFactorization_add_mul_nuw_mul_var(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @tryFactorization_add_mul_nuw_mul_var(
; CHECK-NEXT:    [[MUL21:%.*]] = add i32 [[Y:%.*]], [[Z:%.*]]
; CHECK-NEXT:    [[ADD1:%.*]] = mul i32 [[X:%.*]], [[MUL21]]
; CHECK-NEXT:    ret i32 [[ADD1]]
;
  %mul1 = mul nuw i32 %x, %y
  %mul2 = mul nuw i32 %x, %z
  %add1 = add i32 %mul1, %mul2
  ret i32 %add1
}
