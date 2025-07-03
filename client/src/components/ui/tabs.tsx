import * as React from "react"
import { cn } from "@/lib/utils"

const TabsContext = React.createContext<{
    value: string
    onValueChange: (value: string) => void
} | null>(null)

const Tabs = React.forwardRef<
    HTMLDivElement,
    React.HTMLAttributes<HTMLDivElement> & {
        value: string
        onValueChange: (value: string) => void
    }
>(({ className, value, onValueChange, ...props }, ref) => (
    <TabsContext.Provider value={{ value, onValueChange }}>
        <div ref={ref} className={cn("w-full", className)} {...props} />
    </TabsContext.Provider>
))
Tabs.displayName = "Tabs"

const TabsList = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    ({ className, ...props }, ref) => (
        <div
            ref={ref}
            className={cn(
                "inline-flex h-10 items-center justify-center rounded-md bg-gray-100 p-1 text-gray-500",
                className
            )}
            {...props}
        />
    )
)
TabsList.displayName = "TabsList"

const TabsTrigger = React.forwardRef<
    HTMLButtonElement,
    React.ButtonHTMLAttributes<HTMLButtonElement> & {
        value: string
    }
>(({ className, value: triggerValue, ...props }, ref) => {
    const context = React.useContext(TabsContext)
    if (!context) {
        throw new Error("TabsTrigger must be used within a Tabs component")
    }
    const { value, onValueChange } = context

    return (
        <button
            ref={ref}
            className={cn(
                "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
                value === triggerValue
                    ? "bg-white text-gray-950 shadow-sm"
                    : "text-gray-500 hover:text-gray-900",
                className
            )}
            onClick={() => onValueChange(triggerValue)}
            {...props}
        />
    )
})
TabsTrigger.displayName = "TabsTrigger"

const TabsContent = React.forwardRef<
    HTMLDivElement,
    React.HTMLAttributes<HTMLDivElement> & {
        value: string
    }
>(({ className, value: contentValue, ...props }, ref) => {
    const context = React.useContext(TabsContext)
    if (!context) {
        throw new Error("TabsContent must be used within a Tabs component")
    }
    const { value } = context

    if (value !== contentValue) {
        return null
    }

    return (
        <div
            ref={ref}
            className={cn(
                "mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2",
                className
            )}
            {...props}
        />
    )
})
TabsContent.displayName = "TabsContent"

export { Tabs, TabsList, TabsTrigger, TabsContent }
