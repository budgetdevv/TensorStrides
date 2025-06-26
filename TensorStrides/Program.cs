using System;
using System.Buffers;
using System.Numerics.Tensors;
using FluentAssertions;

namespace TensorStrides
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            int[] arr = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ];

            var arrLength = arr.Length;

            const int NUM_BATCHES = 100;

            // Here, we emulate a larger dimension without allocating a larger array.
            // var tensorSpan = new TensorSpan<int>(
            //     arr,
            //     lengths: [ NUM_BATCHES, arr.Length ],
            //     strides: [ 0, 1 ]
            // );

            var tensorSpan = Tensor.Create<int>(
                arr,
                lengths: [ 1, arrLength ]
            );

            tensorSpan = tensorSpan.Expand([ NUM_BATCHES, arrLength ]);

            var buffer = new int[arrLength];

            var bufferSpan = new Span<int>(buffer);

            for (int i = 0; i < NUM_BATCHES; i++)
            {
                var slice = tensorSpan[i..(i + 1), NRange.All];

                ((int) slice.FlattenedLength).Should().Be(arrLength);

                slice.FlattenTo(bufferSpan);

                Console.WriteLine(bufferSpan.GetSpanPrintString());
            }
        }
    }
}