const { v4: uuidv4 } = require('uuid');
const db = require('../db/database');

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

module.exports = (req, res, next) => {
  let sessionId = req.header('x-session-id');

  // Validate header if provided
  if (sessionId && !UUID_REGEX.test(sessionId)) {
    return res.status(400).json({ success: false, error: 'Invalid session id format' });
  }

  if (!sessionId) {
    // Generate a new UUID when none supplied
    sessionId = uuidv4();
  }

  // Persist (or ensure) user row
  try {
    db.prepare('INSERT OR IGNORE INTO users (session_id) VALUES (?)').run(sessionId);
  } catch (err) {
    // Log but do not block request processing
    console.error('DB error while inserting session', err);
  }

  // Make session ID available to downstream handlers
  req.sessionId = sessionId;
  // Echo it back to the client so they can cache it
  res.set('x-session-id', sessionId);

  next();
}; 