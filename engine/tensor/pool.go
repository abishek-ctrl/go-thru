package tensor

// Pool provides reusable workspaces keyed by size class.
// The engine holds one Workspace per loaded model; Pool is for multi-request servers.
type Pool struct {
	ch chan *Workspace
}

// NewPool creates a pool with capacity n.
func NewPool(n int) *Pool {
	return &Pool{ch: make(chan *Workspace, n)}
}

// Put returns a workspace to the pool.
func (p *Pool) Put(ws *Workspace) {
	select {
	case p.ch <- ws:
	default:
		// drop if full
	}
}

// Get retrieves a workspace or returns nil if empty.
func (p *Pool) Get() *Workspace {
	select {
	case ws := <-p.ch:
		return ws
	default:
		return nil
	}
}
