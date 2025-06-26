using System;
using System.Diagnostics.CodeAnalysis;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorStrides
{
    public static class TensorHelpers<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_values")]
        public static extern ref T[] GetValuesArrayRef(Tensor<T> tensor);
    }

    public static class TensorHelpers
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor<T> MutateDimensionsUnsafely<T>(
            Tensor<T> tensor,
            ReadOnlySpan<nint> newDims)
            where T: unmanaged
        {
            var arr = TensorHelpers<T>.GetValuesArrayRef(tensor);

            return Tensor.Create(
                arr,
                newDims,
                tensor.Strides,
                tensor.IsPinned
            );
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor<T> MutateStridesUnsafely<T>(
            Tensor<T> tensor,
            ReadOnlySpan<nint> newStrides)
            where T: unmanaged
        {
            var arr = TensorHelpers<T>.GetValuesArrayRef(tensor);

            return Tensor.Create(
                arr,
                tensor.Lengths,
                newStrides,
                tensor.IsPinned
            );
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tensor<T> MutateUnsafely<T>(
            Tensor<T> tensor,
            ReadOnlySpan<nint> newDims,
            ReadOnlySpan<nint> newStrides)
            where T: unmanaged
        {
            var arr = TensorHelpers<T>.GetValuesArrayRef(tensor);

            return Tensor.Create(
                arr,
                newDims,
                newStrides,
                tensor.IsPinned
            );
        }

        [SkipLocalsInit]
        public static unsafe Tensor<T> Expand<T>(
            this Tensor<T> tensor,
            ReadOnlySpan<nint> newDims)
            where T: unmanaged
        {
            // https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html
            // Note that unlike the PyTorch implementation, it does not accept a dimension of -1.

            var existingDims = tensor.Lengths;

            var newDimsLength = newDims.Length;

            if (existingDims.Length != newDimsLength)
            {
                ThrowNonMatchingDimensions();
            }

            ref var currentExistingDim = ref MemoryMarshal.GetReference(existingDims);

            ref var currentNewDim = ref MemoryMarshal.GetReference(newDims);

            ref var newDimLastOffsetByOne = ref Unsafe.Add(ref currentNewDim, newDimsLength);

            ref var currentExistingStride = ref MemoryMarshal.GetReference(tensor.Strides);

            var newStridesStart = stackalloc nint[newDimsLength];

            var currentNewStride = newStridesStart;

            for (; !Unsafe.AreSame(ref currentNewDim, ref newDimLastOffsetByOne)
                 ; currentExistingDim = ref Unsafe.Add(ref currentExistingDim, 1)
                 , currentNewDim = ref Unsafe.Add(ref currentNewDim, 1)
                 , currentExistingStride = ref Unsafe.Add(ref currentExistingStride, 1)
                 , currentNewStride++)
            {
                if (currentExistingDim == currentNewDim)
                {
                    *currentNewStride = currentExistingStride;
                }

                else
                {
                    if (currentExistingDim == 1)
                    {
                        *currentNewStride = 0;
                    }

                    else
                    {
                        ThrowNonOneDimension();
                    }
                }
            }

            var newStridesSpan = new ReadOnlySpan<nint>(newStridesStart, newDimsLength);

            return MutateUnsafely(
                tensor,
                newDims,
                newStridesSpan
            );

            [DoesNotReturn]
            void ThrowNonMatchingDimensions()
            {
                throw new ArgumentException("New dimensions must match the number of existing dimensions.");
            }

            [DoesNotReturn]
            void ThrowNonOneDimension()
            {
                throw new ArgumentException(
                    "Expanded dimensions must have an initial dimension of 1."
                );
            }
        }
    }
}