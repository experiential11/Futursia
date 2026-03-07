import express from 'express'
import cors from 'cors'
import { spawn } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const app = express()
const PORT = process.env.PORT || 3001

app.use(cors())
app.use(express.json())

let pythonProcess = null
let pythonServer = null

function initPython() {
  return new Promise((resolve, reject) => {
    pythonProcess = spawn('python', [path.join(__dirname, '..', 'server', 'python_bridge.py')], {
      cwd: path.join(__dirname, '..')
    })

    pythonProcess.stdout.on('data', (data) => {
      const message = data.toString()
      console.log(`[Python] ${message}`)
      if (message.includes('ready')) {
        resolve()
      }
    })

    pythonProcess.stderr.on('data', (data) => {
      console.error(`[Python Error] ${data}`)
    })

    pythonProcess.on('error', reject)

    setTimeout(() => {
      resolve()
    }, 2000)
  })
}

function callPython(method, params = {}) {
  return new Promise((resolve, reject) => {
    const request = {
      method,
      params,
      timestamp: Date.now()
    }

    const pythonStdin = JSON.stringify(request) + '\n'

    pythonProcess.stdin.write(pythonStdin, (err) => {
      if (err) {
        reject(err)
        return
      }

      const timeout = setTimeout(() => {
        reject(new Error('Python response timeout'))
      }, 5000)

      const onData = (data) => {
        clearTimeout(timeout)
        pythonProcess.stdout.removeListener('data', onData)

        try {
          const response = JSON.parse(data.toString())
          resolve(response)
        } catch (e) {
          reject(new Error('Invalid JSON response from Python'))
        }
      }

      pythonProcess.stdout.once('data', onData)
    })
  })
}

app.get('/api/movers', async (req, res) => {
  try {
    const result = await callPython('get_top_movers')
    res.json({ movers: result.movers || [] })
  } catch (err) {
    console.error('Error fetching movers:', err)
    res.status(500).json({ error: err.message })
  }
})

app.get('/api/quote/:symbol', async (req, res) => {
  try {
    const result = await callPython('get_quote', { symbol: req.params.symbol })
    res.json(result)
  } catch (err) {
    console.error('Error fetching quote:', err)
    res.status(500).json({ error: err.message })
  }
})

app.get('/api/forecast/:symbol', async (req, res) => {
  try {
    const result = await callPython('get_forecast', { symbol: req.params.symbol })
    res.json(result)
  } catch (err) {
    console.error('Error fetching forecast:', err)
    res.status(500).json({ error: err.message })
  }
})

app.get('/api/bars/:symbol', async (req, res) => {
  try {
    const limit = req.query.limit || 240
    const result = await callPython('get_bars', { symbol: req.params.symbol, limit: parseInt(limit) })
    res.json(result)
  } catch (err) {
    console.error('Error fetching bars:', err)
    res.status(500).json({ error: err.message })
  }
})

app.get('/api/news/:symbol', async (req, res) => {
  try {
    const limit = req.query.limit || 10
    const result = await callPython('get_news', { symbol: req.params.symbol, limit: parseInt(limit) })
    res.json(result)
  } catch (err) {
    console.error('Error fetching news:', err)
    res.status(500).json({ error: err.message })
  }
})

app.get('/api/config', async (req, res) => {
  try {
    const result = await callPython('get_config')
    res.json(result)
  } catch (err) {
    console.error('Error fetching config:', err)
    res.status(500).json({ error: err.message })
  }
})

process.on('SIGINT', () => {
  console.log('Shutting down...')
  if (pythonProcess) {
    pythonProcess.kill()
  }
  process.exit(0)
})

async function start() {
  try {
    console.log('Initializing Python bridge...')
    await initPython()

    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`)
      console.log(`Frontend will connect to http://localhost:5173`)
    })
  } catch (err) {
    console.error('Failed to start server:', err)
    process.exit(1)
  }
}

start()
