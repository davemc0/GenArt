#include "Interop.h"

#include "CUDAHelpers.h"
#include "Individual.h"
#include "UI.h"

#include "GL/glew.h"

// This needs to come after GLEW
#include "GL/freeglut.h"

#include "cuda_gl_interop.h"
#include "cuda_runtime_api.h"
//#include "device_types.h"

uc4DImage::uc4DImage()
{
    m_vboCUDA = NULL;
    m_vbo = 0;
    m_pitch = 0;
    m_renderTime = -1.0f;
    m_spoofed_uc4Image = true;
    m_launched = false;

    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
}

uc4DImage::~uc4DImage()
{
    // std::cerr << "~uc4DImage(): " << m_vbo << "\n";

    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);

    if (m_vboCUDA) cudaGraphicsUnregisterResource(m_vboCUDA);
    if (m_vbo) glDeleteBuffers(1, &m_vbo);

    if (m_spoofed_uc4Image) uc4Image::clear(); // It was a spoofed pointer, so don't delete it; just unhook it. Was: SetImage().
}

void uc4DImage::SetSize(const int wid_ /*= 0*/, const int hgt_ /*= 0*/, const bool doFill /*= false*/, const bool doDel /*= true*/)
{
    m_vbo = 0;
    GL_ASSERT();
    glGenBuffers(1, &m_vbo);
    GL_ASSERT();
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, 4 * wid_ * hgt_, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GL_ASSERT();

    cudaGraphicsGLRegisterBuffer(&m_vboCUDA, m_vbo, cudaGraphicsMapFlagsNone);
    checkCUDAError("cudaGraphicsGLRegisterBuffer");

    // std::cerr << "m_vboCUDA = " << m_vboCUDA << '\n';

    m_spoofed_uc4Image = true;
    uc4Image::SetImage(reinterpret_cast<uc4Pixel*>(0xdeadbeefull), wid_, hgt_, false); // Not a pointer. We only want the VBO, not the pointer, for now.
    m_pitch = w() * sizeof(uc4Image::PixType);
}

const uc4Pixel* uc4DImage::pp(const int i /*=0*/)
{
    int wid = w();
    int hgt = h();
    ASSERT_D(wid > 0 && hgt > 0);

    if (m_spoofed_uc4Image) static_cast<uc4Image*>(this)->clear(); // Unhook the spoofed image, if any
    m_spoofed_uc4Image = false;

    // Do a real host allocation and copy the image data from device
    if (empty()) {
        static_cast<uc4Image*>(this)->SetSize(wid, hgt);

        uc4Pixel* dp = static_cast<uc4Pixel*>(map());
        cudaMemcpy(static_cast<uc4Image*>(this)->pp(), dp, size_bytes(), cudaMemcpyDeviceToHost);
        unmap();
    }

    return uc4Image::pp(i);
}

void* uc4DImage::map()
{
    cudaGraphicsMapResources(1, &m_vboCUDA, 0);
    checkCUDAError("cudaGraphicsMapResources");

    unsigned char* pixels = NULL;
    size_t num_bytes = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&pixels, &num_bytes, m_vboCUDA);
    checkCUDAError("cudaGraphicsResourceGetMappedPointer");
    ASSERT_R(num_bytes == size_bytes());

    return pixels;
}

void uc4DImage::unmap()
{
    cudaGraphicsUnmapResources(1, &m_vboCUDA, 0);
    checkCUDAError("cudaGraphicsUnmapResources");
}

void uc4DImage::renderTimerStart()
{
    cudaEventRecord(m_start, 0);
}

void uc4DImage::renderTimerStop()
{
    cudaEventRecord(m_stop, 0);
}

bool uc4DImage::renderTimerFinished() const
{
    cudaError_t res = cudaEventQuery(m_stop);
    return (res == ::cudaSuccess) && !empty(); // cudaSuccess makes sure it's done; !empty() makes sure the event was recorded.
}

void uc4DImage::renderBlock()
{
    cudaEventSynchronize(m_stop);
}

float uc4DImage::renderTime()
{
    if (m_renderTime < 0 && renderTimerFinished()) {
        cudaEventSynchronize(m_stop); // This would block if we hadn't made sure it was done.
        cudaEventElapsedTime(&m_renderTime, m_start, m_stop);
    }

    return m_renderTime;
}

bool uc4DImage::imDone() const
{
    return renderTimerFinished();
}

bool uc4DImage::needsLaunch() const
{
    return !m_launched;
}

void uc4DImage::setLaunched(bool val)
{
    m_launched = val;
}
