//
//  PayneSmoothing.swift
//  DepthLiveness
//
//  Created by Giles Payne on 2021/04/01.
//

import Foundation
import opencv2

let OpenCVErrorDomain = "OpenCVErrorDomain"

enum OpenCVError : Int {
  case InvalidKernelSize = 10002
}

func throwInvalidKernelSize(size: Int32) throws {
    throw NSError(
        domain: OpenCVErrorDomain,
        code: OpenCVError.InvalidKernelSize.rawValue,
        userInfo: [
            NSLocalizedDescriptionKey: "Invalid kernal size \(size)"
        ]
    )
}

//the Payne kernel will "score" high for points in a speck and low for points outside a speck
func makePayneKernel(_ size: Int32) throws -> Mat {
    if size % 2 != 1 {
        try throwInvalidKernelSize(size: size)
    }
    let mat = Mat(rows: size, cols: size, type: CvType.CV_32F)
    var totalElements:Float = 0
    var buffer = [Float](repeating: 0, count: Int(size * size));
    for i in 0..<size {
        for j in 0..<size {
            let elementVal = sqrt(Float((size/2 - i) * (size/2 - i) + (size/2 - j) * (size/2 - j)))
            buffer[Int(i * size + j)] = elementVal
            totalElements += elementVal
        }
    }
    try! mat.put(row: 0, col: 0, data: buffer)
    Core.multiply(src1: mat, src2: Mat.ones(size: mat.size(), type: mat.type()), dst: mat, scale: Double(1 / totalElements))
    return mat
}

let KERNEL_SIZE:Int32 = 5

func payneSpeckRemoval(_ fileName: String, _ rows: Int32, _ cols: Int32) -> (Mat, Mat) {
    //read the data in a Mat of UInt16
    let data = NSData(contentsOfFile: fileName)!
    let rawMat = Mat(rows: rows, cols: cols, type: CvType.CV_16U, data: data as Data)

    //make and apply the Payne Kernel
    let convolution = Mat()
    let payneKernel = try! makePayneKernel(KERNEL_SIZE)
    Imgproc.filter2D(src: rawMat, dst: convolution, ddepth: CvType.CV_16U, kernel: payneKernel)

    //make 2 masks - one will contain points in a speck the other points not in a speck
    let mask = Mat()
    Core.compare(src1: convolution, src2: rawMat, dst: mask, cmpop: .CMP_LT)
    let invMask = Mat()
    Core.bitwise_not(src: mask, dst: invMask)

    let outputImage = Mat()
    //apply the mask to get points not in a speck
    Core.bitwise_or(src1: rawMat, src2: rawMat, dst: outputImage, mask: invMask)

    //replace the points in a speck by averaging the surrounding points
    let invMaskedImageConvolution = Mat(), invMaskConvolution = Mat()
    //this gives a weighted sum of points that are nearby a speck but are outside it
    Imgproc.filter2D(src: outputImage, dst: invMaskedImageConvolution, ddepth: CvType.CV_16U, kernel: payneKernel)
    //this gives a sum of the nearby points
    Imgproc.filter2D(src: invMask, dst: invMaskConvolution, ddepth: CvType.CV_16U, kernel: payneKernel)
    let smoothedData = Mat()
    //this gives us the "smoothed" value
    Core.divide(src1: invMaskedImageConvolution, src2: invMaskConvolution, dst: smoothedData, scale: 255.0)
    
    //combine the points that were in a speck and that were smoothed with points that were not in a speck and left as is
    smoothedData.copy(to: outputImage, mask: mask)
    return (rawMat, outputImage)
}

enum Visualization {
    case saturateCast
    case normalizeMinMax
    case equalizeHistogram
}

func outputNirDataAsGrayscaleImage(_ nirData: Mat, _ fileName: String, _ visualization: Visualization = .normalizeMinMax) {
    let nirData8bit = Mat()
    switch visualization {
        case .saturateCast:
            //elements bigger than 255 will be rounded down to 255
            nirData.convert(to: nirData8bit, rtype: CvType.CV_8U)
        case .normalizeMinMax:
            //linearly scales all values between the min and max to fit range 0 to 255
            Core.normalize(src: nirData, dst: nirData8bit, alpha: 0, beta: 255, norm_type: .NORM_MINMAX, dtype: CvType.CV_8U)
        case .equalizeHistogram:
            //redistributes the range of values such that they are spread roughly equally over the range 0 to 255
            nirData.convert(to: nirData8bit, rtype: CvType.CV_8U , alpha: 1.0/256.0)
            Imgproc.equalizeHist(src: nirData8bit, dst: nirData8bit)
    }
    Imgcodecs.imwrite(filename: fileName, img: nirData8bit)
}
