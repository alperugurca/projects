import { useState } from 'react'
import { Terminal, Play, Trash2, Copy, Check } from 'lucide-react'
import './App.css'

type Command = {
  id: string
  name: string
  command: string
  description: string
}

function App() {
  const [commands, setCommands] = useState<Command[]>([
    { id: '1', name: 'List Files', command: 'ls -la', description: 'List all files with details' },
    { id: '2', name: 'Current Directory', command: 'pwd', description: 'Show current working directory' },
    { id: '3', name: 'System Info', command: 'uname -a', description: 'Display system information' },
    { id: '4', name: 'Disk Usage', command: 'df -h', description: 'Show disk space usage' },
  ])
  
  const [newCommand, setNewCommand] = useState({ name: '', command: '', description: '' })
  const [output, setOutput] = useState<string>('')
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const executeCommand = (cmd: string) => {
    setOutput(`Executing: ${cmd}\n${'> '.repeat(20)}\nCommand output would appear here...`)
  }

  const addCommand = () => {
    if (newCommand.name && newCommand.command) {
      const command: Command = {
        id: Date.now().toString(),
        ...newCommand
      }
      setCommands([...commands, command])
      setNewCommand({ name: '', command: '', description: '' })
    }
  }

  const deleteCommand = (id: string) => {
    setCommands(commands.filter(cmd => cmd.id !== id))
  }

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  return (
    <div className="app">
      <header className="header">
        <Terminal size={32} />
        <h1>Terminal Runner</h1>
        <p className="subtitle">Minimalist command executor</p>
      </header>

      <main className="main">
        <div className="container">
          <div className="commands-section">
            <h2>Saved Commands</h2>
            <div className="commands-grid">
              {commands.map((cmd) => (
                <div key={cmd.id} className="command-card">
                  <div className="command-header">
                    <h3>{cmd.name}</h3>
                    <div className="command-actions">
                      <button 
                        className="icon-btn" 
                        onClick={() => copyToClipboard(cmd.command, cmd.id)}
                        title="Copy command"
                      >
                        {copiedId === cmd.id ? <Check size={16} /> : <Copy size={16} />}
                      </button>
                      <button 
                        className="icon-btn run-btn"
                        onClick={() => executeCommand(cmd.command)}
                        title="Run command"
                      >
                        <Play size={16} />
                      </button>
                      <button 
                        className="icon-btn delete-btn"
                        onClick={() => deleteCommand(cmd.id)}
                        title="Delete command"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                  <code className="command-text">{cmd.command}</code>
                  {cmd.description && (
                    <p className="command-description">{cmd.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="add-command-section">
            <h2>Add New Command</h2>
            <div className="form">
              <input
                type="text"
                placeholder="Command name (e.g., 'List Files')"
                value={newCommand.name}
                onChange={(e) => setNewCommand({...newCommand, name: e.target.value})}
                className="input"
              />
              <input
                type="text"
                placeholder="Terminal command (e.g., 'ls -la')"
                value={newCommand.command}
                onChange={(e) => setNewCommand({...newCommand, command: e.target.value})}
                className="input"
              />
              <input
                type="text"
                placeholder="Description (optional)"
                value={newCommand.description}
                onChange={(e) => setNewCommand({...newCommand, description: e.target.value})}
                className="input"
              />
              <button 
                onClick={addCommand}
                className="add-btn"
                disabled={!newCommand.name || !newCommand.command}
              >
                Add Command
              </button>
            </div>
          </div>

          <div className="output-section">
            <h2>Output</h2>
            <div className="output-terminal">
              <pre>{output || 'Run a command to see output here...'}</pre>
            </div>
            <div className="quick-actions">
              <button className="action-btn" onClick={() => setOutput('')}>
                Clear Output
              </button>
              <button className="action-btn" onClick={() => executeCommand('clear')}>
                Clear Terminal
              </button>
            </div>
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>Terminal Runner • Minimal & Efficient</p>
      </footer>
    </div>
  )
}

export default App
