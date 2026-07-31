# React Hooks

Hooks let you use different React features from your components. You can either use the built-in Hooks or combine them to build your own.

## State Hooks

State lets a component "remember" information like user input.

### useState

Declares a state variable that you can update directly.

```javascript
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  );
}
```

### useReducer

Declares a state variable with the update logic inside a reducer function.

```javascript
import { useReducer } from 'react';

function reducer(state, action) {
  switch (action.type) {
    case 'increment': return { count: state.count + 1 };
    default: return state;
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, { count: 0 });
  return <button onClick={() => dispatch({ type: 'increment' })}>{state.count}</button>;
}
```

## Effect Hooks

### useEffect

Connects a component to an external system.

```javascript
import { useEffect } from 'react';

function Timer() {
  useEffect(() => {
    const id = setInterval(() => console.log('tick'), 1000);
    return () => clearInterval(id);
  }, []);
}
```

## Context Hooks

### useContext

Reads and subscribes to a context.

```javascript
const value = useContext(MyContext);
```
