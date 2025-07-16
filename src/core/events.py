"""
Event System for Inter-Component Communication

Publish-subscribe event system enabling loose coupling between application components.
Supports synchronous and asynchronous event handling with priority and filtering.
"""

import asyncio
import logging
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
from concurrent.futures import ThreadPoolExecutor
import uuid


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """
    Event data structure for inter-component communication.
    
    Attributes:
        type: Event type identifier
        data: Event payload data
        source: Source component identifier
        timestamp: Event creation timestamp
        priority: Event priority level
        event_id: Unique event identifier
        tags: Optional event tags for filtering
    """
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Validate event after initialization."""
        if not self.type:
            raise ValueError("Event type cannot be empty")
        
        if not isinstance(self.data, dict):
            raise ValueError("Event data must be a dictionary")


@dataclass
class EventSubscription:
    """
    Event subscription information.
    
    Attributes:
        handler: Event handler function
        event_types: Set of event types to handle
        subscription_id: Unique subscription identifier
        priority: Handler priority (higher numbers execute first)
        async_handler: Whether handler is async
        filters: Optional event filters
        max_events: Maximum events to handle (None = unlimited)
        events_handled: Number of events handled
        created_at: Subscription creation timestamp
    """
    handler: Callable
    event_types: Set[str]
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    async_handler: bool = False
    filters: Optional[Dict[str, Any]] = None
    max_events: Optional[int] = None
    events_handled: int = 0
    created_at: float = field(default_factory=time.time)
    
    def matches_event(self, event: Event) -> bool:
        """Check if this subscription matches an event."""
        # Check event type
        if event.type not in self.event_types and '*' not in self.event_types:
            return False
        
        # Check max events
        if self.max_events and self.events_handled >= self.max_events:
            return False
        
        # Check filters
        if self.filters:
            for key, value in self.filters.items():
                if key == 'source' and event.source != value:
                    return False
                elif key == 'priority' and event.priority != value:
                    return False
                elif key == 'tags' and not (set(value) & event.tags):
                    return False
                elif key in event.data and event.data[key] != value:
                    return False
        
        return True


class EventHandler:
    """Base class for event handlers."""
    
    def __init__(self, handler_id: str = None):
        self.handler_id = handler_id or str(uuid.uuid4())
        self.subscriptions: List[EventSubscription] = []
    
    def handle_event(self, event: Event) -> Any:
        """Handle an event. Override in subclasses."""
        pass
    
    async def handle_event_async(self, event: Event) -> Any:
        """Handle an event asynchronously. Override in subclasses."""
        return self.handle_event(event)


class EventBus:
    """
    Central event bus for publish-subscribe communication.
    
    Provides thread-safe event publishing and subscription with support for
    synchronous and asynchronous handlers, event filtering, and priority handling.
    """
    
    def __init__(self, max_queue_size: int = 10000, thread_pool_size: int = 4):
        self.subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.global_subscriptions: List[EventSubscription] = []
        self.event_queue: deque = deque(maxlen=max_queue_size)
        self.event_history: deque = deque(maxlen=1000)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.async_loop = None
        
        self._lock = threading.RLock()
        self._shutdown = False
        self._stats = {
            'events_published': 0,
            'events_handled': 0,
            'handlers_executed': 0,
            'errors': 0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Start async event loop in separate thread
        self._start_async_loop()
    
    def subscribe(self, event_types: Union[str, List[str]], handler: Callable,
                 priority: int = 0, filters: Dict[str, Any] = None,
                 max_events: int = None) -> EventSubscription:
        """
        Subscribe to events.
        
        Args:
            event_types: Event type(s) to subscribe to
            handler: Handler function
            priority: Handler priority (higher executes first)
            filters: Optional event filters
            max_events: Maximum events to handle
            
        Returns:
            EventSubscription: Subscription object
        """
        if isinstance(event_types, str):
            event_types = {event_types}
        else:
            event_types = set(event_types)
        
        # Determine if handler is async
        async_handler = asyncio.iscoroutinefunction(handler)
        
        subscription = EventSubscription(
            handler=handler,
            event_types=event_types,
            priority=priority,
            async_handler=async_handler,
            filters=filters,
            max_events=max_events
        )
        
        with self._lock:
            # Add to global subscriptions if subscribing to all events
            if '*' in event_types:
                self.global_subscriptions.append(subscription)
                self.global_subscriptions.sort(key=lambda s: s.priority, reverse=True)
            else:
                # Add to specific event type subscriptions
                for event_type in event_types:
                    self.subscriptions[event_type].append(subscription)
                    self.subscriptions[event_type].sort(key=lambda s: s.priority, reverse=True)
        
        self.logger.debug(f"Subscribed to events {event_types} with handler {handler.__name__}")
        return subscription
    
    def unsubscribe(self, subscription: EventSubscription) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription: Subscription to remove
            
        Returns:
            bool: True if successfully unsubscribed
        """
        try:
            with self._lock:
                # Remove from global subscriptions
                if subscription in self.global_subscriptions:
                    self.global_subscriptions.remove(subscription)
                
                # Remove from specific event type subscriptions
                for event_type in subscription.event_types:
                    if subscription in self.subscriptions[event_type]:
                        self.subscriptions[event_type].remove(subscription)
                
                self.logger.debug(f"Unsubscribed {subscription.subscription_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe {subscription.subscription_id}: {e}")
            return False
    
    def publish(self, event: Event) -> bool:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
            
        Returns:
            bool: True if event was published successfully
        """
        if self._shutdown:
            return False
        
        try:
            with self._lock:
                # Add to queue and history
                self.event_queue.append(event)
                self.event_history.append(event)
                self._stats['events_published'] += 1
            
            # Process event immediately
            self._process_event(event)
            
            self.logger.debug(f"Published event {event.type} from {event.source}")
            return True
            
        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(f"Failed to publish event {event.type}: {e}")
            return False
    
    def publish_async(self, event: Event) -> bool:
        """
        Publish an event asynchronously.
        
        Args:
            event: Event to publish
            
        Returns:
            bool: True if event was queued for publishing
        """
        if self._shutdown:
            return False
        
        try:
            # Submit to thread pool for processing
            self.thread_pool.submit(self.publish, event)
            return True
            
        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(f"Failed to publish async event {event.type}: {e}")
            return False
    
    def get_subscriptions(self, event_type: str = None) -> List[EventSubscription]:
        """
        Get subscriptions for an event type.
        
        Args:
            event_type: Event type (None for all subscriptions)
            
        Returns:
            List[EventSubscription]: List of subscriptions
        """
        with self._lock:
            if event_type:
                return list(self.subscriptions[event_type])
            else:
                all_subs = list(self.global_subscriptions)
                for subs in self.subscriptions.values():
                    all_subs.extend(subs)
                return all_subs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            return {
                **self._stats,
                'queue_size': len(self.event_queue),
                'subscription_count': sum(len(subs) for subs in self.subscriptions.values()) + len(self.global_subscriptions),
                'event_types': list(self.subscriptions.keys())
            }
    
    def clear_history(self):
        """Clear event history."""
        with self._lock:
            self.event_history.clear()
    
    def shutdown(self):
        """Shutdown the event bus."""
        self._shutdown = True
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True, timeout=5.0)
        
        # Close async loop
        if self.async_loop and not self.async_loop.is_closed():
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
        
        self.logger.info("Event bus shutdown complete")
    
    def _process_event(self, event: Event):
        """Process an event by calling all matching handlers."""
        handlers_to_call = []
        
        with self._lock:
            # Get global handlers
            for subscription in self.global_subscriptions:
                if subscription.matches_event(event):
                    handlers_to_call.append(subscription)
            
            # Get specific event type handlers
            for subscription in self.subscriptions[event.type]:
                if subscription.matches_event(event):
                    handlers_to_call.append(subscription)
        
        # Sort by priority
        handlers_to_call.sort(key=lambda s: s.priority, reverse=True)
        
        # Execute handlers
        for subscription in handlers_to_call:
            try:
                if subscription.async_handler:
                    # Submit async handler to async loop
                    if self.async_loop and not self.async_loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            subscription.handler(event),
                            self.async_loop
                        )
                else:
                    # Execute sync handler in thread pool
                    self.thread_pool.submit(self._execute_handler, subscription, event)
                
                subscription.events_handled += 1
                self._stats['handlers_executed'] += 1
                
            except Exception as e:
                self._stats['errors'] += 1
                self.logger.error(f"Error executing handler {subscription.subscription_id}: {e}")
        
        self._stats['events_handled'] += 1
    
    def _execute_handler(self, subscription: EventSubscription, event: Event):
        """Execute a synchronous event handler."""
        try:
            subscription.handler(event)
        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(f"Handler {subscription.subscription_id} failed: {e}")
    
    def _start_async_loop(self):
        """Start async event loop in separate thread."""
        def run_async_loop():
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()


# Global event bus instance
_global_event_bus: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def create_event_bus(max_queue_size: int = 10000, thread_pool_size: int = 4) -> EventBus:
    """
    Create a new event bus instance.
    
    Args:
        max_queue_size: Maximum size of event queue
        thread_pool_size: Size of thread pool for handlers
        
    Returns:
        EventBus: New event bus instance
    """
    return EventBus(max_queue_size, thread_pool_size)


def get_global_event_bus() -> EventBus:
    """Get or create the global event bus instance."""
    global _global_event_bus
    
    with _event_bus_lock:
        if _global_event_bus is None:
            _global_event_bus = create_event_bus()
        return _global_event_bus


def set_global_event_bus(event_bus: EventBus):
    """Set the global event bus instance."""
    global _global_event_bus
    
    with _event_bus_lock:
        if _global_event_bus:
            _global_event_bus.shutdown()
        _global_event_bus = event_bus


# Convenience functions using global event bus
def publish_event(event_type: str, data: Dict[str, Any] = None, source: str = None,
                 priority: EventPriority = EventPriority.NORMAL, tags: Set[str] = None) -> bool:
    """
    Publish an event using the global event bus.
    
    Args:
        event_type: Event type identifier
        data: Event payload data
        source: Source component identifier
        priority: Event priority level
        tags: Optional event tags
        
    Returns:
        bool: True if event was published successfully
    """
    event = Event(
        type=event_type,
        data=data or {},
        source=source,
        priority=priority,
        tags=tags or set()
    )
    
    return get_global_event_bus().publish(event)


def subscribe_to_event(event_types: Union[str, List[str]], handler: Callable,
                      priority: int = 0, filters: Dict[str, Any] = None,
                      max_events: int = None) -> EventSubscription:
    """
    Subscribe to events using the global event bus.
    
    Args:
        event_types: Event type(s) to subscribe to
        handler: Handler function
        priority: Handler priority
        filters: Optional event filters
        max_events: Maximum events to handle
        
    Returns:
        EventSubscription: Subscription object
    """
    return get_global_event_bus().subscribe(event_types, handler, priority, filters, max_events)


def unsubscribe(subscription: EventSubscription) -> bool:
    """
    Unsubscribe from events using the global event bus.
    
    Args:
        subscription: Subscription to remove
        
    Returns:
        bool: True if successfully unsubscribed
    """
    return get_global_event_bus().unsubscribe(subscription)


# Decorator for event handlers
def event_handler(event_types: Union[str, List[str]], priority: int = 0,
                 filters: Dict[str, Any] = None, max_events: int = None):
    """
    Decorator for registering event handlers.
    
    Args:
        event_types: Event type(s) to handle
        priority: Handler priority
        filters: Optional event filters
        max_events: Maximum events to handle
        
    Returns:
        Function decorator
    """
    def decorator(func):
        subscription = subscribe_to_event(event_types, func, priority, filters, max_events)
        func._event_subscription = subscription
        return func
    
    return decorator