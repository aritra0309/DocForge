# useState

`useState` is a React Hook that lets you add a state variable to your component.

```javascript
const [state, setState] = useState(initialState)
```

## Reference

### useState(initialState)

Call `useState` at the top level of your component to declare a state variable.

#### Parameters

\*\*initialState\*\*
: The value you want the state to be initially. It can be a value of any type, but there is a special behavior for functions.

#### Returns

`useState` returns an array with exactly two values:

1. The current state.
2. The set function that lets you update the state to a different value and trigger a re-render.

### set functions, like setSomething(nextState)

The set function returned by useState lets you update the state to a different value and trigger a re-render.

```javascript
const [name, setName] = useState('Edward');

function handleClick() {
  setName('Taylor');
}
```

## Usage

### Adding state to a component

```javascript
import { useState } from 'react';

function MyComponent() {
  const [age, setAge] = useState(28);
  const [name, setName] = useState('Taylor');
  const [todos, setTodos] = useState(() => createTodos());
  // ...
}
```

### Updating state based on the previous state

```javascript
function handleClick() {
  setAge(a => a + 1); // setAge(42 => 43)
  setAge(a => a + 1); // setAge(43 => 44)
  setAge(a => a + 1); // setAge(44 => 45)
}
```
